// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * TaskAllocation Smart Contract
 * Stores MEC task allocation decisions on-chain.
 * Compatible with PBFT consortium blockchain model.
 */
contract TaskAllocation {

    struct Task {
        uint256 taskId;
        uint256 serverId;
        uint256 unitPrice;
        uint256 dataSize;
        uint256 timeConstraint;
        uint256 utilityReward;
        uint256 timestamp;
        bool completed;
    }

    struct AllocationRound {
        uint256 roundId;
        uint256 totalUtility;
        uint256 completedTasks;
        uint256 timestamp;
        address submittedBy;
    }

    mapping(uint256 => Task) public tasks;
    mapping(uint256 => AllocationRound) public rounds;
    uint256 public taskCount;
    uint256 public roundCount;
    address public admin;

    event TaskAllocated(uint256 indexed taskId, uint256 serverId, uint256 utilityReward);
    event RoundCompleted(uint256 indexed roundId, uint256 totalUtility);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    function logAllocation(
        uint256 taskId,
        uint256 serverId,
        uint256 unitPrice,
        uint256 dataSize,
        uint256 timeConstraint,
        uint256 utilityReward
    ) public onlyAdmin {
        tasks[taskId] = Task({
            taskId: taskId,
            serverId: serverId,
            unitPrice: unitPrice,
            dataSize: dataSize,
            timeConstraint: timeConstraint,
            utilityReward: utilityReward,
            timestamp: block.timestamp,
            completed: true
        });
        taskCount++;
        emit TaskAllocated(taskId, serverId, utilityReward);
    }

    function logRound(uint256 roundId, uint256 totalUtility, uint256 completedTasks) public onlyAdmin {
        rounds[roundId] = AllocationRound({
            roundId: roundId,
            totalUtility: totalUtility,
            completedTasks: completedTasks,
            timestamp: block.timestamp,
            submittedBy: msg.sender
        });
        roundCount++;
        emit RoundCompleted(roundId, totalUtility);
    }

    function getTask(uint256 taskId) public view returns (Task memory) {
        return tasks[taskId];
    }

    function getRound(uint256 roundId) public view returns (AllocationRound memory) {
        return rounds[roundId];
    }

    function getTotalRounds() public view returns (uint256) {
        return roundCount;
    }
}
