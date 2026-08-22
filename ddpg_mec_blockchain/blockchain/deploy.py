"""
Deploy TaskAllocation contract to local Ganache testnet.
Run: python blockchain/deploy.py
Requires: ganache running on port 8545 (ganache --port 8545)
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web3 import Web3
from solcx import compile_source, install_solc
from config import BLOCKCHAIN_CONFIG


def deploy_contract():
    install_solc("0.8.0")

    sol_path = os.path.join(os.path.dirname(__file__), "contracts", "TaskAllocation.sol")
    with open(sol_path, "r") as f:
        source = f.read()

    compiled = compile_source(source, output_values=["abi", "bin"], solc_version="0.8.0")
    contract_id, contract_interface = compiled.popitem()
    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]

    w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_CONFIG["ganache_url"]))
    assert w3.is_connected(), "Cannot connect to Ganache. Start: ganache --port 8545"

    account = w3.eth.accounts[0]
    w3.eth.default_account = account

    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = Contract.constructor().transact({"gas": BLOCKCHAIN_CONFIG["gas_limit"]})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    deployment_info = {
        "address": receipt.contractAddress,
        "abi": abi,
        "deployer": account,
        "block": receipt.blockNumber,
    }

    out_path = os.path.join(os.path.dirname(__file__), "deployment.json")
    with open(out_path, "w") as f:
        json.dump(deployment_info, f, indent=2)

    print(f"Contract deployed at: {receipt.contractAddress}")
    return deployment_info


if __name__ == "__main__":
    deploy_contract()
