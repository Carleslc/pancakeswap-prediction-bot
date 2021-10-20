from typing import Callable, Union
from web3.types import ABI, ABIFunction
from eth_typing.evm import ChecksumAddress

import os

from dotenv import load_dotenv

from web3 import Web3
from bscscan import BscScan
from bscscan.utils.conversions import Conversions
from decimal import Decimal

from utils import error

from web3.contract import Contract, ContractFunction, ContractFunctions
from web3._utils.abi import abi_to_signature, get_abi_input_types, get_abi_input_names, get_abi_output_types

# BSCScan API
# Limits: 5 calls/second, up to 100.000 calls/day
# https://docs.bscscan.com/
# https://docs.bscscan.com/support/rate-limits

class BinanceSmartChain:

  MAIN_NET = 'https://bsc-dataseed.binance.org'

  def __init__(self, api_key: str, address: str, debug=False):
    # https://web3py.readthedocs.io/en/stable/web3.main.html#web3-api
    self._web3 = Web3(Web3.HTTPProvider(BinanceSmartChain.MAIN_NET))
    # https://bscscan-python.pankotsias.com/ 
    self._scan = BscScan(api_key, asynchronous=False, debug=debug)
    self._address = self._web3.toChecksumAddress(address)
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
    return self.info['Proxy'] != '0'
  
  @property
  def is_contract(self) -> bool:
    return True if self._contract is not None else self._web3.eth.get_code(self._address) != '0x'
  
  @property
  def functions(self) -> ContractFunctions:
    return self.contract.functions
  
  def get_bnb_balance(self):
    return from_wei(self._scan.get_bnb_balance(self._address))

  def get_contract_abi(self) -> ABI:
    if self._contract:
      return self._contract.abi
    # https://docs.bscscan.com/api-endpoints/contracts#get-contract-abi-for-verified-contract-source-codes
    # https://api.bscscan.com/api?module=contract&action=getabi&address=ADDRESS&apikey=
    # https://bscscan-python.pankotsias.com/bscscan.modules.html#bscscan.modules.contracts.Contracts.get_contract_abi
    return self._scan.get_contract_abi(self._address)
  
  def get_contract_info(self) -> list:
    if self._info is None:
      # https://docs.bscscan.com/api-endpoints/contracts#get-contract-source-code-for-verified-contract-source-codes
      # https://api.bscscan.com/api?module=contract&action=getsourcecode&address=ADDRESS&apikey=
      # https://bscscan-python.pankotsias.com/bscscan.modules.html#bscscan.modules.contracts.Contracts.get_contract_source_code
      self._info = self._scan.get_contract_source_code(self._address)
    return self._info

  def get_contract_circulating_supply(self) -> float:
    return float(self._scan.get_circulating_supply_by_contract_address(self._address))

  def list_functions(self, with_names: bool = True) -> list[str]:
    f_info = function_info if with_names else function_signature
    return [f_info(f) for f in self.contract.all_functions()]
  
  def call(self, f: ContractFunction, *args, **kwargs) -> dict:
    results = f.call(*args, **kwargs)
    if type(results) is not list:
      results = [results]
    return dict(zip(get_abi_output_names(f.abi), results))
  
  def _check_contract(self):
    if self._contract is None and self.is_contract:
      self._contract = self._get_contract()

  def _get_contract(self) -> Contract:
    contract = self._web3.eth.contract(address=self._address, abi=self.abi)
    self._add_methods(contract)
    return contract
  
  def _add_methods(self, contract: Contract):
    for f in contract.all_functions():
      setattr(self, str(f.function_identifier), wrap_call(f))

  def __call__(self, data: str) -> str:
    return self._scan.get_proxy_call(to=self._address, data=data)

  def __enter__(self):
    self._scan.__enter__()
    self._check_contract()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._scan.__exit__(exc_type, exc_val, exc_tb)

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

def wrap_call(f: ContractFunction, *call_args, **call_kwargs) -> Callable:
  def call(*args, **kwargs):
    return f(*args, **kwargs).call(*call_args, **call_kwargs)
  return call

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
