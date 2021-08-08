#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "layers.h"

using std::vector;
using std::string;

namespace ntwrk {
	class Network {
	public:
		int depth = -1;
		Input* inputLayer;
		Dense* outputLayer;
		Loss* lossFunc;
		vector<float> weights, biases, widths;
		vector<string> activations;
		vector<vector<int>> inpLayers;
		void Compile(Loss* lossFunc);
		void SetInput(Input* layer);
		void AddLayer(Dense* layer);
		void Train(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize, bool shuffle = true);
		void SaveModel(string fileName);
		void LoadModel(string fileName);
		void Test(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs);
		vector<float> Predict(vector<float> example);
		private:
		void RunEpochs(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize);
		void ShuffleBatch(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs);
		void OutputProg(int progress, int amount, float cost);
		//void BackPropegate(vector<vector<float>> desiredOutputs);
	};
}