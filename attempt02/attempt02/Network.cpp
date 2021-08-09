#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <numeric>

#include <direct.h>

#include "Network.h"
#include "Layers.h"
#include "node.h"
#include "Activations.h"

using namespace std;
using namespace ntwrk;

void Network::SetInput(int width) {
	widths.push_back(width);
	inpIndexes.push_back({});
	activations.push_back("");
	depth++;
}

/*void Network::SetInput(Input* layer) {
	depth++;
	inputLayer = layer;
	layer->AddNodes({});
}*/

void Network::AddLayer(int width, vector<int> _inpLayers, string _activationName) {
	widths.push_back(width);
	inpIndexes.push_back(_inpLayers);
	activations.push_back(_activationName);
	depth++;
}

/*void Network::AddLayer(Dense* layer) {
	depth++;
	layer->index = depth;
	inputLayer->AddLayer(layer);
	layer->AddNodes();
	outputLayer = layer;
}*/

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
	weights.push_back({});
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

/*void Network::SaveModel(string fileName) {
	string fileName = "/" + fileName;// + ".txt";
	const char* dir = fileName.c_str();
	Network& ref = *this;
	bool saved;

	saved = mkdir(dir);

	fstream file("network", ios::out);
	// Writing the object's data in file
	file.write((char*)&ref, sizeof(ref));
	//cout << "saved model" << endl;
	inputLayer->SaveLayer(fileName);
}*/

void Network::LoadModel(string fileName) {
	std::ifstream file;
	Network& temp = *this;
	// Opening file in input mode
	file.open(fileName, ios::in);

	// Reading from file into object "obj"
	file.read((char*)&temp, sizeof(temp));
	cout << "loaded model" << endl;
}

void Network::SetLoss(string lossFuncNam) {
	lossFuncName = lossFuncNam;
	if (lossFuncNam == "BinCrossEntro") {
		BinCrossEntro* a = new BinCrossEntro;
		lossFunc = a;
	}
	else if (lossFuncNam == "CatCrossEntro") {
		CatCrossEntro* a = new CatCrossEntro;
		lossFunc = a;
	}
	else if (lossFuncNam == "MeanSquareError") {
		MeanSquareError* a = new MeanSquareError;
		lossFunc = a;
	}
}

void Network::Compile(string _lossFunc) {
	// = *inputLayer;
	inputLayer = new Input(widths[0]);
	//vector<vector<float>>* weightsPointer = &weights;
	inputLayer->AddNodes();
	for (int i = 1; i < depth; i++) {
		Dense* layer = new Dense(widths[i], inpIndexes[i], activations[i]);
		inputLayer->AddLayer(layer);
		layer->index = i;
		layer->AddNodes();
		outputLayer = layer;
	}
	SetLoss(_lossFunc);
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

void Network::PopulateWeights() {
	vector<float>& Ws = weights;
	inputLayer->SumWeights(Ws);
}

void Network::SaveModel(string fileName) {
	//PopulateWeights();

	PopulateWeights();

	fileName += ".txt";
	//Network& network = *this;

	Network temp;
	temp.weights = weights;
	temp.lossFuncName = lossFuncName;
	temp.inpIndexes = inpIndexes;
	temp.activations = activations;
	temp.biases = biases;

	ofstream file;

	file.open(fileName, ios::out | ios::trunc);
	// Writing the object's data in file
	file.write((char*)&temp, sizeof(temp));

	//file.write((char*)this, sizeof(*this));
	file.close();
	cout << "saved model" << endl;
}