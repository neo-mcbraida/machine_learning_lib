#pragma once
// node.h
#include <iostream>
#include <vector>

using std::vector;

namespace ntwrk {
	class Node {
	public:
		float activation, biasChange = 0, bias = 0;
		vector<float> weights, sumWBChanges, desiredVals;
		vector<Node*> inpNodes;
		Node(vector<Node*> nodes);
		void SetActivation();
		void AdjustWB(int batchSize);
		void SetPassChanges(float derivActivation);
	private:
		float GetAveDerCost();
		void SetPrevNodeDesire(float avDerCost, int nodeInd, float derivActivation);
		void SetWeightGrad(float avDerCost, int nodeInd, float derivActivation);
		void SetBiasGrad(float avDerCost, float derivActivation);
		float RandomVal();
	};
}