#pragma once
// node.h
#include <iostream>
#include <vector>

using std::vector;

namespace ntwrk {
	class Node {
	public://EwrtX is error with respect to rawVal
		float activation, rawVal, EwrtX = 0, bias = 0;
		vector<float> sumWBChanges, desiredVals;
		vector<float> weights = {};
		vector<Node*> inpNodes;
		Node(vector<Node*>, int);
		void SetActivation();
		void AdjustWB(int batchSize);
		//void SaveNode();
	private:
		float RandomVal();
	};

	class MemoryNode {
	public:
		vector<float> history;
	};
}


vector<int> tSeriesDat = { 0, 13, 9, 4, 8, 12, 13, 31, 35, 8, 73, 77, 21, 104, 105, 92, 192, 2, 67, 283, 135, 129, 203, 152, 107, 185, 110, 138, 100, 228, 130, 137, 235, 138, 166, 122, 149, 124, 187, 195, 130, 126, 154, 149, 106, 151 ,100, 91, 46, 38, 13, 22, 33, 9, 14, 48, 7, 18, 6, 21, 30, 21, 79, 64, 55, 18, 101, 63, 59, 6, 26, 150, 80, 82, 58, 84, 59, 60, 116, 99, 112, 197, 116, 215, 82, 155, 80, 13, 70, 391, 178, 124, 235, 88, 187, 140, 111, 72, 75, 44, 40, 37, 10, 6, 7, 5, 10, 11, 16, 20, 17, 21, 34, 28, 44, 33, 85, 16, 95, 43, 35, 30, 259 }