{-# LANGUAGE BangPatterns #-}

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
    , Epoch
    , Verbosity(..)

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

    -- * Running the neural network
    , runNN

    -- * Training the neural network
    , trainUntil
    , trainNTimes

    -- * Temporary
    , test
    ) where

import           Control.Monad                   (forM, replicateM)
import           Control.Monad.Primitive         (PrimMonad, PrimState)
import           Numeric.LinearAlgebra.HMatrix   (Matrix, Vector, (><), vector, (#>), cmap, tr, scale, asColumn, asRow, (<>), (|>), size, sumElements)
import           System.Random.MWC               (Gen, asGenST,
                                                  withSystemRandom)
import           System.Random.MWC.Distributions (normal, standard)
import Data.Default
import Data.List (foldl', scanl')
import Data.List.Split (chunksOf)
import Debug.Trace (trace)
import System.IO (hSetBuffering, stdout, BufferMode(..))

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
type Bias                    = Vector Double
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

-- |Defines an epoch number, useful for signatures (in predicates for instance)
type Epoch = Int

-- |The sigmoid activation function
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp(-x))

-- |The derivatives of the sigmoid activation function
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

-- |Allows for the training to report is progress
-- Please note that they are inclusive (latters include formers)
data Verbosity = Quiet
               | EpochNumber
               | CostOnEpoch
               | CostOnBatch
               deriving (Eq,Show,Ord)

-- |Creates a neural network. Ensures it is valid or fail
mkNeuralNetwork :: InitStrategy -> ActivationFunction -> ActivationFunctionDeriv -> Structure -> IO NeuralNetwork
mkNeuralNetwork _ _ _ struct | length struct < 2 = error "A neural network must have at least 2 layers (input and output)"
                             | any (<= 0) struct = error "You can't have zero or a negative number of neurons in a layer"
mkNeuralNetwork NguyenWidrow _ _ _ =
    error "The Nguyen-Widrow random weight initialization method is not yet implemented, either use basic or normalized"
mkNeuralNetwork initStrat h h' struct = do
    weights <- forM (zip struct (tail struct)) $ \(n1, n2) -> do
        randomNumbers <- withSystemRandom . asGenST $ \gen -> replicateM (n1*n2) (distrib n1 gen)
        return $ (n2><n1) randomNumbers
    biases <- forM (tail struct) $ \n -> do
        randomNumbers <- withSystemRandom . asGenST $ \gen -> replicateM n (standard gen)
        return (vector randomNumbers)
    return NeuralNetwork {
          nnStructure = struct
        , nnWeights = weights
        , nnBiases = biases
        , nnActivation = h
        , nnActivationDeriv = h'
    }
    where distrib :: PrimMonad m => Int -> Gen (PrimState m) -> m Double
          distrib n1 = case initStrat of
              Basic      -> standard -- standard generate a mean 0 and variance 1
              Normalized -> normal 0 (1 / sqrt (fromIntegral n1))
              _          -> error "mkNeuralNetwork > distrib: unsupported InitStrategy"

-- |Runs the Neural Network on the input to get the output
runNN :: NeuralNetwork -> Vector Double -> Vector Double
runNN nn input = foldl' (\a (w,b) -> cmap σ (w #> a + b)) input (zip weights biases)
    where weights = nnWeights nn
          biases  = nnBiases nn
          σ       = nnActivation nn

-- |Trains (with backpropagation) the Neural Network until the predicate returns true
trainUntil :: (Epoch -> NeuralNetwork -> TrainingSet -> Bool) -> Parameters -> NeuralNetwork -> TrainingSet -> NeuralNetwork
trainUntil predicate params nn trainingSet = train nn 0
    where train :: NeuralNetwork -> Epoch -> NeuralNetwork
          train nn' !epoch | predicate epoch nn' trainingSet = nn'
                           | otherwise                       = let newNN = backProp epoch params nn' trainingSet
                                                               in train newNN (epoch+1)

-- |Performs one epoch of backpropagation on the Neural Network
backProp :: Epoch -> Parameters -> NeuralNetwork -> TrainingSet -> NeuralNetwork
backProp epoch params !nn trainingSet =
    let (newWeights, newBiases) = foldl' (trainOnBatch epoch) (nnWeights nn, nnBiases nn) batches
    -- in nn {nnWeights = newWeights, nnBiases = newBiases}
        !newNN = nn {nnWeights = newWeights, nnBiases = newBiases}

        -- outputs = map (runNN newNN) (map fst trainingSet)
        -- !cost = (1.0 / (2.0 * fromIntegral (length trainingSet)) *) $ sum $ zipWith (\output target -> cmap (^2) (output - target)) outputs (map snd trainingSet)
    in newNN
    where batches = chunksOf batchSize trainingSet

          batchSize = case pBackpropStrategy params of
            BatchGradientDescent       -> length trainingSet
            MiniBatchGradientDescent n -> n
            OnlineGradientDescent      -> 1
            --_                          -> error "backProp: unsupported Backprop strategy"

          -- Accumulates gradient (error) on all examples in a batch
          trainOnBatch :: Epoch -> ([WeightMatrix], [Bias]) -> TrainingSet -> ([WeightMatrix], [Bias])
          trainOnBatch epoch (weights, biases) batch =
              -- Create empty matrices and vectors with same size as the weights and biases, all initialized to 0
              let (initialNabla_W, initialNabla_B) = (map (scale 0) weights, map (scale 0) biases)
                  (nabla_W, nabla_B) = foldl' trainOnExample (initialNabla_W, initialNabla_B) batch

                  updatedWeights = zipWith (\w nW -> w - scale (η / m) nW) weights nabla_W
                  updatedBiases  = zipWith (\b nB -> b - scale (η / m) nB) biases nabla_B
            --   in (updatedWeights, updatedBiases)
                  !newNN = nn {nnWeights = updatedWeights, nnBiases = updatedBiases}
                  outputs = map (runNN newNN) (map fst trainingSet)
                  !cost = (1.0 / (2.0 * fromIntegral (length trainingSet)) *) $ sum $ zipWith (\output target -> cmap (^2) (output - target)) outputs (map snd trainingSet)

              in trace ("#" ++ show epoch ++ " - " ++ show cost) $ (updatedWeights, updatedBiases)
              where
                -- Computes gradient (error) on one example of the batch
                trainOnExample :: ([Matrix Double], [Vector Double]) -> TrainingExample -> ([Matrix Double], [Vector Double])
                trainOnExample (accNabla_W, accNabla_B) (input, target) =
                    -- Performs feed-forward, keeping both zs and as
                    -- Input layer has no z value, so the initial left value in the tuple doesn't matter
                    let forwardPass = scanl' feedForward (size input |> repeat 0.0, input) (zip weights biases)

                    -- Computes the output error (last layer only)
                        (zL, aL) = last forwardPass
                        outputError = derivCost aL target * cmap σ' zL -- Hadamard product

                    -- Backpropagates the error
                    -- Don't take the first weight matrice, because we don't backpropagate the error to the input layer
                    -- Don't take the last forwardPass because it is alreayd included in the initial accumulator
                    -- don't take the first forwardPass because we don't backpropagate the error to the input layer
                        δs = scanr backpropagate outputError (zip (tail weights) (map fst ((tail . init) forwardPass)))

                    -- Accumulates the gradient
                        accNabla_B' = zipWith (+) δs accNabla_B
                        accNabla_W' = zipWith3 (\δ a nW -> nW + (asColumn δ <> asRow a) ) δs (map snd (init forwardPass)) accNabla_W

                    in (accNabla_W', accNabla_B')

                -- Performs the feed-forward pass, keeping both z and a values
                feedForward :: (Vector Double, Vector Double) -> (WeightMatrix, Bias) -> (Vector Double, Vector Double)
                feedForward (_, a) (w, b) = let newZ = (w #> a) + b
                                                newA = cmap σ newZ
                                            in (newZ, newA)

                -- This is the gradient of the cost function
                derivCost :: Vector Double -> Vector Double -> Vector Double
                derivCost output target = case pCostFunction params of
                    MeanSquare   -> target - output
                    CrossEntropy -> error "backProp > trainOnBatch > trainOnExample > derivCost: Cross entropy function not handled yet"
                    --_            -> error "backProp > trainOnBatch > trainOnExample > derivCost: unsupported cost function"

                -- Backpropagates the error among the layer
                backpropagate :: (WeightMatrix, Vector Double) -> Vector Double -> Vector Double
                backpropagate (w, z) δ = ((tr w) #> δ) * cmap σ' z -- Hadamard product

                -- Extracts the learning rate from the parameters
                η :: Double
                η = case pLearningRateStrategy params of
                        FixedRate η' -> η'
                        --_            -> error "backProp > trainOnBatch > trainOnExample > η: unsupported learning rate strategy"

                -- Computes the length of the batch
                m :: Double
                m = (fromIntegral . length) batch

          σ = nnActivation nn
          σ' = nnActivationDeriv nn

-- |Convenience function: trains for a fixed number of times
trainNTimes :: Int -> Parameters -> NeuralNetwork -> TrainingSet -> NeuralNetwork
trainNTimes n | n < 0     = error "trainNTimes: cannot train a negative number of times"
              | otherwise = trainUntil (\epoch _ _ -> epoch == n)

-- Convenience function: trains the network until the error reaches a threshold (limits the number of epochs)
{-
trainUntilErrorBelow :: Double -> Int -> Parameters -> NeuralNetwork -> TrainingSet -> NeuralNetwork
trainUntilErrorBelow errThreshold maxEpochs
    | maxEpochs < 0    = error "trainUntilErrorBelow: the maximum number of epochs cannot be negative"
    | errThreshold < 0 = error "trainUntilErrorBelow: the error threshold cannot be negative"
    | otherwise        = trainUntil predicate
-}

test :: IO ()
test = do
    hSetBuffering stdout NoBuffering
    nn <- mkNeuralNetwork def sigmoid sigmoid' [2,2,1]
    let input = [ vector [0,0]
                , vector [0,1]
                , vector [1,0]
                , vector [1,1]
                ] :: [Vector Double]
        targets = [ vector [0]
                  , vector [1]
                  , vector [1]
                  , vector [0]
                  ] :: [Vector Double]
        trainingSet = zip input targets

    putStrLn "Initial output: "
    mapM_ (print . asColumn . runNN nn) input
    let params = Parameters {
                       pLearningRateStrategy = FixedRate 3.0
                     , pBackpropStrategy = BatchGradientDescent
                     , pCostFunction = MeanSquare
                 }

        newNN = trainNTimes 1000 params nn trainingSet

    putStrLn "\n===================\nFinal output: "
    mapM_ (print . asColumn . runNN newNN) input
