from typing import Callable, Union
from web3.types import ABI, ABIFunction
from eth_typing.evm import AnyAddress, ChecksumAddress
from hexbytes.main import HexBytes

import os
from time import sleep

from dotenv import load_dotenv

from web3 import Web3
from bscscan import BscScan
from bscscan.utils.conversions import Conversions
from decimal import Decimal

from utils import error

from web3.contract import Contract, ContractFunction, ContractFunctions, ACCEPTABLE_EMPTY_STRINGS
from web3._utils.abi import abi_to_signature, get_abi_input_types, get_abi_input_names, get_abi_output_types
from web3.exceptions import ContractLogicError

# BSCScan API
# https://docs.bscscan.com/
# https://docs.bscscan.com/support/rate-limits
# Limits: 5 calls/second, up to 100.000 calls/day
MAX_BSC_SCAN_API_CALLS_PER_SECOND = 5

def _ratelimit(max_calls_per_second: int = MAX_BSC_SCAN_API_CALLS_PER_SECOND):
  sleep(1/max_calls_per_second)

def ratelimit(f):
  def decorator(*args, **kwargs):
    _ratelimit()
    return f(*args, **kwargs)
  return decorator

class BinanceSmartChain:

  MAIN_NET = 'https://bsc-dataseed.binance.org'

  _IMPLEMENTATION_SLOT = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'

  def __init__(self, api_key: str, address: str, resolve_proxies=True, debug=False):
    # https://web3py.readthedocs.io/en/stable/web3.main.html#web3-api
    self._web3 = Web3(Web3.HTTPProvider(BinanceSmartChain.MAIN_NET))
    # https://bscscan-python.pankotsias.com/ 
    self._scan = BscScan(api_key, asynchronous=False, debug=debug)
    self._address = self._implementation = self.checksum(address)
    self._resolve_proxies = resolve_proxies
    self._debug = debug
    self._contract = None
    self._info = None
  
  @property
  def contract(self) -> Contract:
    return self._contract
  
  @property
  def address(self) -> ChecksumAddress:
    return self._address
  
  @property
  def abi(self) -> ABI:
    return self.get_contract_abi()
  
  @property
  def info(self):
    return self.get_contract_info()[0]
  
  @property
  def source(self) -> str:
    return self.info['SourceCode']
  
  @property
  def name(self) -> str:
    return self.info['ContractName']
  
  @property
  def is_proxy(self) -> bool:
    return self._is_proxy_address(self._address)
  
  @property
  def is_contract(self) -> bool:
    return True if self._contract is not None else self._is_contract_address(self._address)
  
  @property
  def functions(self) -> ContractFunctions:
    return self.contract.functions
  
  @ratelimit
  def get_bnb_balance(self):
    return from_wei(self._scan.get_bnb_balance(self._address))

  def get_contract_abi(self) -> ABI:
    if self._contract:
      return self._contract.abi
    # https://docs.bscscan.com/api-endpoints/contracts#get-contract-abi-for-verified-contract-source-codes
    # https://api.bscscan.com/api?module=contract&action=getabi&address=ADDRESS&apikey=
    # https://bscscan-python.pankotsias.com/bscscan.modules.html#bscscan.modules.contracts.Contracts.get_contract_abi
    _ratelimit()
    return self._scan.get_contract_abi(self._address)
  
  def get_contract_info(self) -> list:
    if self._info is None:
      # https://docs.bscscan.com/api-endpoints/contracts#get-contract-source-code-for-verified-contract-source-codes
      # https://api.bscscan.com/api?module=contract&action=getsourcecode&address=ADDRESS&apikey=
      # https://bscscan-python.pankotsias.com/bscscan.modules.html#bscscan.modules.contracts.Contracts.get_contract_source_code
      _ratelimit()
      self._info = self._scan.get_contract_source_code(self._address)
    return self._info

  @ratelimit
  def get_contract_circulating_supply(self) -> float:
    return float(self._scan.get_circulating_supply_by_contract_address(self._address))
  
  def list_functions(self, with_names: bool = True) -> list[str]:
    f_info = function_info if with_names else function_signature
    return [f_info(f) for f in self.contract.all_functions()]
  
  def _is_contract_address(self, address: str) -> bool:
    return self._web3.eth.get_code(address) not in ACCEPTABLE_EMPTY_STRINGS

  def _is_proxy_address(self, address: str) -> bool:
    if address == self._address:
      info = self.get_contract_info()
    else:
      _ratelimit()
      info = self._scan.get_contract_source_code(address)
    return info[0]['Proxy'] != '0'
  
  def resolve_proxy(self, override: bool = False):
    try:
      implementation = self.contract.get_function_by_signature('implementation()')().call()
    except (ValueError, ContractLogicError):
      implementation = hexBytesToAddress(self._web3.eth.get_storage_at(self._address, BinanceSmartChain._IMPLEMENTATION_SLOT))
    if implementation:
      self._implementation = self.checksum(implementation)
      if override:
        self._address = self._implementation
        self._contract = None
        self._info = None
        self._set_contract()
      else:
        _ratelimit()
        self._contract = self._get_contract(self._address, self._scan.get_contract_abi(self._implementation))
        self._add_methods(self._contract)

  def resolve_proxies(self, override: bool = False):
    while self._is_proxy_address(self._implementation):
      self.resolve_proxy(override)
  
  def call(self, f: ContractFunction, *args, **kwargs) -> Union[tuple, dict]:
    results = f.call(*args, **kwargs)
    if type(results) is not tuple:
      results = (results)
    output_names = get_abi_output_names(f.abi)
    if len(output_names) != len(results):
      return results
    return dict(zip(output_names, results))
  
  def decode_input(self, input: str) -> tuple[ContractFunction, dict]:
    return self.contract.decode_function_input(input)
  
  def _set_contract(self):
    if self._contract is None and self.is_contract:
      self._contract = self._get_contract(self._address)
      self._add_methods(self._contract)
      if self._resolve_proxies:
        self.resolve_proxies()

  def _get_contract(self, address: str, abi: ABI = None) -> Contract:
    if abi is None:
      _ratelimit()
      abi = self._scan.get_contract_abi(address)
    return self._web3.eth.contract(address=address, abi=abi)
  
  def _add_methods(self, contract: Contract):
    for f in contract.all_functions():
      setattr(self, str(f.function_identifier), self.wrap_call(f))

  @ratelimit
  def __call__(self, data: str) -> str:
    return self._scan.get_proxy_call(to=self._address, data=data)

  def __enter__(self):
    self._scan.__enter__()
    self._set_contract()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._scan.__exit__(exc_type, exc_val, exc_tb)
  
  def checksum(self, address: str) -> ChecksumAddress:
    return self._web3.toChecksumAddress(address)
  
  def wrap_call(self, f: ContractFunction, *call_args, **call_kwargs) -> Callable:
    def call(*args, **kwargs):
      f_bound = f(*args, **kwargs)
      if self._debug:
        print(f_bound.selector, f_bound)
      return f_bound.call(*call_args, **call_kwargs)
    return call

def from_wei(value: Union[str, int], decimals: int = 18) -> Decimal:
  return Conversions.to_ticker_unit(int(value), decimals)

def to_wei(value: Union[str, int], decimals: int = 18) -> Decimal:
  return Conversions.to_smallest_unit(int(value), decimals)

def function_signature(f: ContractFunction) -> str:
  return abi_to_signature(f.abi) if f.abi else str(f.function_identifier)

def function_info(f: ContractFunction) -> str:
  def arg_str(arg) -> str:
    return f'{arg[0]}: {arg[1]}' if arg[0] else arg[1]
  def get_abi_inputs() -> list[str]:
    input_names = get_abi_input_names(f.abi)
    input_types = get_abi_input_types(f.abi)
    return list(map(arg_str, zip(input_names, input_types)))
  def get_abi_outputs() -> list[str]:
    output_names = get_abi_output_names(f.abi)
    output_types = get_abi_output_types(f.abi)
    return list(map(arg_str, zip(output_names, output_types)))
  abi_inputs = get_abi_inputs()
  abi_outputs = get_abi_outputs()
  fn_input_args = ', '.join(abi_inputs)
  fn_output_args = ', '.join(abi_outputs)
  if len(abi_outputs) > 1:
    fn_output_args = f'[{fn_output_args}]'
  if len(abi_outputs) > 0:
    fn_output_args = ' -> ' + fn_output_args
  return "{fn_name}({fn_input_args}){fn_output_args}".format(
        fn_name=f.abi['name'],
        fn_input_args=fn_input_args,
        fn_output_args=fn_output_args
    )

def get_abi_output_names(abi: ABIFunction) -> list[str]:
  if 'outputs' not in abi and abi['type'] == 'fallback':
      return []
  else:
      return [arg['name'] for arg in abi['outputs']]

def hexBytesToAddress(hexbytes: HexBytes) -> str:
  return HexBytes(hexbytes.hex()[-40:]).hex()

def same_address(address1: AnyAddress, address2: AnyAddress) -> bool:
  return HexBytes(address1) == HexBytes(address2)

BSC_CLIENT = None

def bsc_client(address: str, debug: bool = False) -> BinanceSmartChain:
  global BSC_CLIENT

  if not BSC_CLIENT:
    print(f"Loading {BinanceSmartChain.__name__} API...")

    load_dotenv()

    api_key = os.getenv('BSCSCAN_API_KEY')

    if not api_key:
      error("Missing BSCSCAN_API_KEY (.env)")
    
    BSC_CLIENT = lambda address, debug: BinanceSmartChain(api_key, address, debug=debug)

  return BSC_CLIENT(address, debug)
