#pragma once
// node.h
#include <iostream>
#include <vector>
#include "Activations.h"

using std::vector;

namespace ntwrk {
	class LSTMNode : public Node{
	public:
		vector<Node*> inpNodes;
		//vector<vector<float>> weightsF, weightsI, weightsA, weightsO;
		//vector<vector<float>> weightChangeF, weightChangeI, weightChaOngeA, weightChangeO;
		vector<vector<float>> inputs;//
		vector<vector<float>> sums;
		//vector<float> Us;
		vector<vector<float>> weights;// in order F I A O
		vector<vector<float>> weightChange;
		//vector<float> UChange;
		vector<float> biases;
		Sigmoid sigmoid;
		Tanh tanH;
		vector<float> Fz, Iz, Az, Oz;// z denotes zero
		vector<float> state, out;
		vector<float> totalErrors;
		LSTMNode(vector<Node*>);
		void SetActivation();
		float ReturnSum(vector<float> weights, float bias);
		void SetState();
		void SetOut();
		float GetStateByPrevState(int index);
		float GetActivationByState(int index);
		float GetNextStateByState(int index);
		void BackProp(float nextState, float nextForget, float NextErrorbyActiv);
		void AddWeightChange(int time, float nextState, float common_deriv);
		float GetAByWo(float sum, float input, int index);
		float GetCByWf(float sum, float input, int index);
		float GetCByWu(float sum, float input, int index);
		float GetCByWc(float sum, float input, int index);
		float GetPrevError(float common_deriv, int index);
	private:
		float RandomVal();
	};
}