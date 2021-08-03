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
		//Network();
		void SetInput(Input* layer);
		void AddLayer(Dense* layer);
		void Train(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize, bool shuffle = true);
		//void SaveModel(string fileName);
		//void LoadModel(string fileName);
		//void TestModel(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs);
		//void Predict(vector<vector<float>> example);
		private:
		void RunEpochs(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize);
		void ShuffleBatch(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs);
		void OutputProg(int progress, int amount, float cost);
		//void BackPropegate(vector<vector<float>> desiredOutputs);
	};
}