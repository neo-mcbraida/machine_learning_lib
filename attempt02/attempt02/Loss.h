#pragma once
#include <vector>

using std::vector;

namespace ntwrk {
	class Loss {
	public:
		virtual float GetNodesLoss(float, vector<float>) = 0;
		virtual float GetDerLoss(float, vector<float>) = 0;
	};
	
	class BinCrossEntro : public Loss {
	public:
		virtual float GetNodesLoss(float, vector<float>);
		virtual float GetDerLoss(float, vector<float>);
	};

	class CatCrossEntro : public Loss {
	public:
		virtual float GetNodesLoss(float, vector<float>);
		virtual float GetDerLoss(float, vector<float>);
	};

	class MeanSquareError : public Loss {
	public:
		virtual float GetNodesLoss(float, vector<float>);
		virtual float GetDerLoss(float, vector<float>);
	};
}