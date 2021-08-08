#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "node.h"
#include "Activations.h"
#include "Loss.h";

using std::vector;
using std::string;
using ntwrk::Node;
using ntwrk::ActivationFunc;

namespace ntwrk {
	class Layer {
	public:
		float totalError = 0;
		int index;
		vector<Node*> nodes;
		ActivationFunc* activation;
		Layer* nextLayer = nullptr;
		Layer* prevLayer = nullptr;
		vector<int> inputLayers;
		int width;
		Layer(int width, string activation);
		virtual void AddNodes(vector<int> inp);
		void AddLayer(Layer* newLayer);
		void EndBackProp();
		virtual void ForwardProp();
		virtual void BackProp();
		virtual void SetPrevEwrtR();
		virtual void SetChanges(int batchSize);
		vector<Node*> GetInpNodes(vector<int> inputInds);
	private:
		void SetActivation(string activation);
	};

	class Dense : public Layer {
	public:
		vector<Layer*> inputLayers;
		vector<int> inpIndexes;
		Dense(int width, vector<int> inpIndexes, string activation);
		virtual void ForwardProp();
		virtual void AddNodes();
		virtual void SetPrevEwrtA();
		void StartBackProp(vector<float> desiredOut, Loss* lossFunc);
		void BackProp();
		float GetCost(vector<float> desiredOut);
		virtual void SetChanges(int batchSize);
	private:
	};
	
	class Input : public Layer{
	public:
		Input(int width);
		void StartForwardProp(vector<float> input);
	private:
		void SetNodes(vector<float> input);
	};
}