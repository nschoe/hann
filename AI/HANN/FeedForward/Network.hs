module AI.HANN.FeedForward.Network
    (
    -- * Types
      NeuralNetwork
    , Parameters(..)
    , LearningRateStrategy(..)
    , BackpropStrategy
    , WeightMatrix
    , Bias
    , CostFunction(..)
    , TrainingExample
    , TrainingSet

    -- * Neural Network accessors
    , getStructure
    , getWeights
    , getBiases
    , getActivation
    , getActivationDeriv

    -- * Activation functions and derivatives
    , sigmoid
    , sigmoid'
    ) where

import Numeric.LinearAlgebra.HMatrix (Matrix, Vector)

-- |Core type of the library, describes a neural network's architecture
data NeuralNetwork = NeuralNetwork {
    -- ^Describes the internal architecture.
    -- [I,H1,H2,O] describes a 4-layered neural network with:
    -- I input units
    -- H1 neurons in the first hidden layer
    -- H2 neurons in the second hidden layer
    -- O output neurons
      nnStructure :: [Int]

    -- ^The neural network's weights
    -- Note that nnWeights[l] is the weight matrix from layer l to layer (l+1)
    , nnWeights :: [WeightMatrix]

    -- ^The units' biases
    -- Note that nnBiases[l] is the bias matrix of layer (l+1) (since first layer is input layer and has no biases)
    , nnBiases :: [Bias]
    , nnActivation :: Double -> Double
    , nnActivationDeriv :: Double -> Double
}

-- |Accessors for the @NeuralNetwork@ type
getStructure :: NeuralNetwork -> [Int]
getStructure = nnStructure

getWeights :: NeuralNetwork -> [WeightMatrix]
getWeights = nnWeights

getBiases :: NeuralNetwork -> [Bias]
getBiases = nnBiases

getActivation :: NeuralNetwork -> (Double -> Double)
getActivation = nnActivation

getActivationDeriv :: NeuralNetwork -> (Double -> Double)
getActivationDeriv = nnActivationDeriv

-- |Just type synonyms to make signatures more readable
type WeightMatrix = Matrix Double
type Bias = Matrix Double

-- |Regroups all parameters for the network
data Parameters = Parameters {
      pLearningRateStrategy :: LearningRateStrategy -- ^Specifies the strategy used to update the learning rate
    , pBackpropStrategy :: BackpropStrategy         -- ^Specifies the gradient descent strategy
    , pCostFunction :: CostFunction                 -- ^Specifies the cost function to use for training
} deriving (Show, Eq)

-- |The strategy to use for udpating the learning rate
data LearningRateStrategy =
    FixedRate Double
    deriving (Show, Eq)

-- |The strategy used for Gradient Descent
data BackpropStrategy =
    -- ^Accumulates error & computes gradient on all training cases before updating weights
      BatchGradientDescent

    -- ^Accumulates error & computes gradient on N training cases, then perform weight update and carry on
    | MiniBatchGradientDescent Int

    -- ^Computes error & gradient and performs weight update after each training case
    | OnlineGradientDescent
    deriving (Show, Eq)

-- |Cost function type
data CostFunction = MeanSquare
                  | CrossEntropy
    deriving (Show, Eq)

-- |First element is the input, second is the expected result (target)
type TrainingExample = (Vector Double, Vector Double)

-- |Defines a collection of training examples
type TrainingSet = [TrainingExample]

-- |The sigmoid activation function
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp(-x))

-- |The derivatives of the sigmoid activation function, expressed in terms of the sigmoid function
sigmoid' :: Double -> Double
sigmoid' x = s * (1 - s)
    where s = sigmoid x
