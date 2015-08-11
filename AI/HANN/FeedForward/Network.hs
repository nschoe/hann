module AI.HANN.FeedForward.Network
    (
    -- * Types
      NeuralNetwork
    , Parameters(..)
    , Structure
    , ActivationFunction
    , ActivationFunctionDeriv
    , LearningRateStrategy(..)
    , BackpropStrategy
    , WeightMatrix
    , Bias
    , CostFunction(..)
    , TrainingExample
    , TrainingSet
    , InitStrategy(..)

    -- * Creating a neural network
    , mkNeuralNetwork

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

import           Control.Monad                   (forM, replicateM)
import           Control.Monad.Primitive         (PrimMonad, PrimState)
import           Numeric.LinearAlgebra.HMatrix   (Matrix, Vector, (><))
import           System.Random.MWC               (Gen, asGenST,
                                                  withSystemRandom)
import           System.Random.MWC.Distributions (normal, standard)
import Data.Default

-- |Core type of the library, describes a neural network's architecture
data NeuralNetwork = NeuralNetwork {
    -- ^Describes the internal architecture.
    -- [I,H1,H2,O] describes a 4-layered neural network with:
    -- I input units
    -- H1 neurons in the first hidden layer
    -- H2 neurons in the second hidden layer
    -- O output neurons
      nnStructure       :: [Int]

    -- ^The neural network's weights
    -- Note that nnWeights[l] is the weight matrix from layer l to layer (l+1)
    , nnWeights         :: [WeightMatrix]

    -- ^The units' biases
    -- Note that nnBiases[l] is the bias matrix of layer (l+1) (since first layer is input layer and has no biases)
    , nnBiases          :: [Bias]
    , nnActivation      :: ActivationFunction
    , nnActivationDeriv :: ActivationFunctionDeriv
}

-- |Accessors for the @NeuralNetwork@ type
getStructure :: NeuralNetwork -> [Int]
getStructure = nnStructure

getWeights :: NeuralNetwork -> [WeightMatrix]
getWeights = nnWeights

getBiases :: NeuralNetwork -> [Bias]
getBiases = nnBiases

getActivation :: NeuralNetwork -> ActivationFunction
getActivation = nnActivation

getActivationDeriv :: NeuralNetwork -> ActivationFunctionDeriv
getActivationDeriv = nnActivationDeriv

-- |Just type synonyms to make signatures more readable
type Structure               = [Int]
type WeightMatrix            = Matrix Double
type Bias                    = Matrix Double
type ActivationFunction      = Double -> Double
type ActivationFunctionDeriv = Double -> Double

-- |Regroups all parameters for the network
data Parameters = Parameters {
      pLearningRateStrategy :: LearningRateStrategy -- ^Specifies the strategy used to update the learning rate
    , pBackpropStrategy     :: BackpropStrategy     -- ^Specifies the gradient descent strategy
    , pCostFunction         :: CostFunction         -- ^Specifies the cost function to use for training
} deriving (Show, Eq)

instance Default Parameters where
    def = Parameters {
      pLearningRateStrategy = def
    , pBackpropStrategy     = def
    , pCostFunction         = def
}

-- |The strategy to use for udpating the learning rate
data LearningRateStrategy =
    FixedRate Double
    deriving (Show, Eq)

instance Default LearningRateStrategy where
    def = FixedRate 1.0

-- |The strategy used for Gradient Descent
data BackpropStrategy =
    -- ^Accumulates error & computes gradient on all training cases before updating weights
      BatchGradientDescent

    -- ^Accumulates error & computes gradient on N training cases, then perform weight update and carry on
    | MiniBatchGradientDescent Int

    -- ^Computes error & gradient and performs weight update after each training case
    | OnlineGradientDescent
    deriving (Show, Eq)

instance Default BackpropStrategy where
    def = MiniBatchGradientDescent 10

-- |Cost function type
data CostFunction = MeanSquare
                  | CrossEntropy
    deriving (Show, Eq)

instance Default CostFunction where
    def = CrossEntropy

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

-- |Different strategies for generating random values for weights and biases
data InitStrategy = Basic         -- ^A gaussian distribution, with mean 0 and stdev 1 (not adapted to big networks)
                  | Normalized    -- ^A gaussian distribution, with mean 0 and stdev 1/√(nbInputNeurons)
                  | NguyenWidrow  -- ^A more sophisticated method, as seen [here](put link)
    deriving (Show, Eq)

instance Default InitStrategy where
    def = Normalized

-- |Creates a neural network. Ensures it is valid or fail
mkNeuralNetwork :: InitStrategy -> ActivationFunction -> ActivationFunctionDeriv -> Structure -> IO NeuralNetwork
mkNeuralNetwork _ _ _ struct | length struct < 2 = error "A neural network must have at least 2 layers (input and output)"
                             | any (<= 0) struct = error "You can't have zero or a negative number of neurons in a layer"
mkNeuralNetwork NguyenWidrow _ _ _ =
    error "The Nguyen-Widrow random weight initialization method is not yet implemented, either use basic or normalized"
mkNeuralNetwork initStrat h h' struct = do
    basicWeights <- forM (zip struct (tail struct)) $ \(n1, n2) -> do
        randomNumbers <- withSystemRandom . asGenST $ \gen -> replicateM (n1*n2) (distrib n1 gen)
        return $ (n2><n1) randomNumbers
    return NeuralNetwork {
          nnStructure = struct
        , nnWeights = basicWeights
        , nnBiases = []
        , nnActivation = h
        , nnActivationDeriv = h'
    }
    where distrib :: PrimMonad m => Int -> Gen (PrimState m) -> m Double
          distrib n1 = case initStrat of
              Basic      -> standard -- standard generate a mean 0 and variance 1
              Normalized -> normal 0 (1 / sqrt (fromIntegral n1))
              _          -> error "mkNeuralNetwork (distrib): unsupported InitStrategy"
