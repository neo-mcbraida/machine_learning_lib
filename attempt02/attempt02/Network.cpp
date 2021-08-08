#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <numeric>
#include "Network.h"
#include "Layers.h"
#include "node.h"
#include "Activations.h"

using namespace std;
using namespace ntwrk;

void Network::SetInput(Input* layer) {
	depth++;
	inputLayer = layer;
	layer->AddNodes({});
}

void Network::AddLayer(Dense* layer) {
	depth++;
	layer->index = depth;
	inputLayer->AddLayer(layer);
	layer->AddNodes();
	outputLayer = layer;
}

void Network::Train(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize, bool shuffle) {
	for (int i = 0; i < epochs; i++) {
		std::cout << "epoch: " << (i + 1) << "/" << epochs << std::endl;
		if (shuffle == true) {
			ShuffleBatch(inputData, desiredOutputs);
		}
		RunEpochs(inputData, desiredOutputs, epochs, batchSize);
	}
}

void Network::RunEpochs(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize) {
	float cost;
	for (int i = 0; i < inputData.size(); i++) {
		vector<float> exampleInp = inputData[i];
		(*inputLayer).StartForwardProp(exampleInp);

		vector<float> goal = desiredOutputs[i];
		outputLayer->StartBackProp(goal, lossFunc);

		if (i % batchSize == 0 && i != 0) {

			for (int u = 0; u < inputLayer->nodes.size(); u++){
				inputLayer->nodes[u]->desiredVals.clear();
			}
			cost = outputLayer->GetCost(goal);
			outputLayer -> SetChanges(batchSize);
			OutputProg(i, inputData.size(), cost);
		}
	}
}

void Network::OutputProg(int progress, int amount, float cost) {
	string output = to_string(progress) + "/" + to_string(amount) + " [";
	float am = (float)amount;
	float prog = (float)progress;
	int numEqs = (int)(round(((float)progress / (float)amount) * 25.0));
	for (int i = 0; i < 25; i++) {
		if (i <= numEqs) {
			output += "#";
		}
		else {
			output += "-";
		}
	}
	output += "] Cost: " + to_string(cost);
	std::cout << output << std::endl;
}

void Network::ShuffleBatch(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs) {
	int seed = unsigned(std::time(0));
	std::srand(seed);
	std::random_shuffle(inputData.begin(), inputData.end());

	std::srand(seed);
	std::random_shuffle(desiredOutputs.begin(), desiredOutputs.end());
}

void Network::SaveModel(string fileName) {
	string fName = fileName + ".txt";
	Network& ref = *this;
	fstream file(fName, ios::out);

	// Writing the object's data in file
	file.write((char*)&ref, sizeof(ref));
	cout << "saved model" << endl;
}

void Network::LoadModel(string fileName) {
	std::ifstream file;
	Network& temp = *this;
	// Opening file in input mode
	file.open(fileName, ios::in);

	// Reading from file into object "obj"
	file.read((char*)&temp, sizeof(temp));
	cout << "loaded model" << endl;
}

void Network::Compile(Loss* _lossFunc) {
	// = *inputLayer;
	lossFunc = _lossFunc;
}

void Network::Test(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs) {
	vector<float> accuracies;
	vector<float> predictions;
	float diff;
	cout << "Testing: " << endl;
	for (int i = 0; i < inputData.size(); i++) {
		predictions = Predict(inputData[i]);
		for (int u = 0; u < desiredOutputs[i].size(); u++) {
			if (desiredOutputs[i][u] == 1) {
				diff = (desiredOutputs[i][u] - predictions[u]) / predictions[u];
				diff = 1 - diff;
				accuracies.push_back(diff);
				break;
			}
		}
		if (i % 32 == 0) {
			float cost = outputLayer->GetCost(desiredOutputs[i]);
			OutputProg(i, inputData.size(), cost);
		}
	}
	float avAccuracy = (accumulate(accuracies.begin(), accuracies.end(), 0)) / inputData.size();
	cout << "accuracy: " << avAccuracy << endl;
}

vector<float> Network::Predict(vector<float> example) {
	(*inputLayer).StartForwardProp(example);
	vector<float> output;
	for (Node* node : outputLayer->nodes) {
		output.push_back(node->activation);
	}
	return output;
}