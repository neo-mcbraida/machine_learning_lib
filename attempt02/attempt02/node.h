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
		void SaveNode();
	private:
		float RandomVal();
	};
}