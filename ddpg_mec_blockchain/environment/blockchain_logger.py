"""
Blockchain logger: records task allocations to smart contract.
Falls back to local logging if blockchain is unavailable.
"""

import json
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLOCKCHAIN_CONFIG


class BlockchainLogger:
    def __init__(self, use_blockchain=None):
        self.use_blockchain = use_blockchain if use_blockchain is not None else BLOCKCHAIN_CONFIG["use_blockchain"]
        self.w3 = None
        self.contract = None
        self.account = None
        self.local_log = []

        if self.use_blockchain:
            self._connect()

    def _connect(self):
        try:
            from web3 import Web3
            self.w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_CONFIG["ganache_url"]))
            if not self.w3.is_connected():
                print("[Blockchain] Ganache not available — falling back to local logging")
                self.use_blockchain = False
                return

            deployment_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "blockchain", "deployment.json"
            )
            with open(deployment_path, "r") as f:
                info = json.load(f)

            self.contract = self.w3.eth.contract(address=info["address"], abi=info["abi"])
            self.account = self.w3.eth.accounts[0]
            self.w3.eth.default_account = self.account
            print(f"[Blockchain] Connected to contract at {info['address']}")
        except Exception as e:
            print(f"[Blockchain] Connection failed: {e} — using local logging")
            self.use_blockchain = False

    def log_task_allocation(self, task_id, server_id, unit_price, data_size, time_constraint, utility):
        record = {
            "task_id": task_id,
            "server_id": server_id,
            "unit_price": unit_price,
            "data_size": data_size,
            "time_constraint": time_constraint,
            "utility": utility,
            "timestamp": time.time(),
        }

        if self.use_blockchain and self.contract:
            try:
                tx = self.contract.functions.logAllocation(
                    int(task_id),
                    int(server_id),
                    int(unit_price * 1000),
                    int(data_size * 1000),
                    int(time_constraint * 1000),
                    int(max(utility, 0) * 1000),
                ).transact({"gas": 200000})
                self.w3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                print(f"[Blockchain] TX failed: {e}")

        self.local_log.append(record)

    def log_round(self, round_id, total_utility, completed_tasks):
        if self.use_blockchain and self.contract:
            try:
                tx = self.contract.functions.logRound(
                    int(round_id),
                    int(max(total_utility, 0) * 100),
                    int(completed_tasks),
                ).transact({"gas": 200000})
                self.w3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                print(f"[Blockchain] Round TX failed: {e}")

    def get_local_log(self):
        return self.local_log
