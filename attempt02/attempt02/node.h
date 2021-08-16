#pragma once
// node.h
#include <iostream>
#include <vector>

using std::vector;

namespace ntwrk {
	class Cell {
	public:
		float activation;
		vector<Cell*> inpNodes;
		virtual void AddError(float error, int index) = 0; 
		//virtual void SetNodeActivation() = 0;
		virtual void AdjustWB(int);
		virtual void SetActivation();
	};

	class Node : public Cell{
	public://EwrtX is error with respect to rawVal
		float rawVal, EwrtX = 0, bias = 0;
		vector<float> sumWBChanges, desiredVals;
		vector<float> weights = {};
		Node(vector<Cell*>);
		virtual void SetActivation();
		void AdjustWB(int batchSize);
		//void SaveNode();
		virtual void AddError(float error, int index);
	private:
		float RandomVal();
	};
}

