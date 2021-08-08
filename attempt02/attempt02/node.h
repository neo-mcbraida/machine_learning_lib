#pragma once
// node.h
#include <iostream>
#include <vector>

using std::vector;

namespace ntwrk {
	class Node {
	public://EwrtX is error with respect to rawVal
		float activation, rawVal, EwrtX = 0, bias = 0;
		vector<float> weights, sumWBChanges, desiredVals;
		vector<Node*> inpNodes;
		Node(vector<Node*> nodes);
		void SetActivation();
		void AdjustWB(int batchSize);
	private:
		float RandomVal();
	};
}