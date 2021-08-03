#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

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
		outputLayer->StartBackProp(goal);

		if (i % batchSize == 0 && i != 0) {

			for (int u = 0; u < inputLayer->nodes.size(); u++){
				inputLayer->nodes[u]->desiredVals.clear();
			}
			string predictions = "";
			for (int i = 0; i < outputLayer->nodes.size(); i++) {
				predictions += to_string(outputLayer->nodes[i]->activation);
				predictions += " ";
			}
			cout << predictions << endl;
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