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
		int depth;
		Input* inputLayer;
		Layer* outputLayer;
		Network();
		void SetInp(Input* layer);
		void AddLayer(Layer* layer);
		void Train(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize, bool shuffle);
		void SaveModel(string fileName);
		void LoadModel(string fileName);
		void TestModel(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs);
		void Predict(vector<vector<float>> example);
		private:
		void RunEpochs(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize);
		void ShuffleBatch(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs);
		void OutputProg(int progress, int amount, float cost);
		void BackPropegate(vector<vector<float>> desiredOutputs);
	};
}