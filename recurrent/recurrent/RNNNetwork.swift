//
//  RNNNetwork.swift
//  Network
//
//  Created by Martin Mumford on 2/20/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

struct RNNDelayedWeight
{
    var value:Double
    var delay:Int
}

struct RNNWeightIndexPair
{
    var startIndex:Int
    var endIndex:Int
}

extension String {
    var doubleValue: Double {
        return (self as NSString).doubleValue
    }
    
    var intValue: Int {
        return (self as NSString).integerValue
    }
}

public class RNNNetwork
{
    // Initialization
    var neuronString:String
    var weightStrings:[String]
    
    var foldedNeurons = [RNNNeuron]()
    var foldedWeights = [Int:[Int:[RNNDelayedWeight]]]()
    
    var unfoldedNeurons = [RNNNeuron]()
    var unfoldedWeights = [Int:[Int:Double]]()
    
    var numberOfBlocks = 0
    
    var learningRate:Double = 0.1
    
    // Training
    var outputs = [Double]()
    var deltas = [Double]()
    var weightDeltas = [Int:[Int:Double]]()
    
    public init(neuronString:String, weightStrings:[String])
    {
        self.neuronString = neuronString
        self.weightStrings = weightStrings
        
        self.resetNetwork()
    }
    
    func resetNetwork()
    {
        // Neuron String Format: number of nodes on each layer, separated by a colon "2:3:4:2"
        // Weight String Format: ["0-1|0.12:1", "1-1|0.5:0"]
        
        ////////////////////////////////////////////////////////////
        // Clear out any pre-existing data
        foldedNeurons.removeAll()
        foldedWeights.removeAll()
        
        unfoldedNeurons.removeAll()
        unfoldedWeights.removeAll()
        
        outputs.removeAll()
        deltas.removeAll()
        weightDeltas.removeAll()
        
        ////////////////////////////////////////////////////////////
        // Populate the folded network (foldedNeurons, foldedWeights)
        let layers:[String] = neuronString.componentsSeparatedByString(":")
        var foldedNeuronIndexCount = 0
        
        for (index:Int, layerString:String) in enumerate(layers)
        {
            if let neuronCountInLayer = layers[index].toInt()
            {
                for neuronIndex in 0..<(neuronCountInLayer)
                {
                    var neuronType:NeuronType = .Hidden
                    
                    if (index == 0)
                    {
                        neuronType = .Input
                    }
                    else if (index == layers.count-1)
                    {
                        neuronType = .Output
                    }
                    
                    addNeuronToFoldedNetwork(foldedNeuronIndexCount, unfoldedIndex:foldedNeuronIndexCount, ioIndex:0, type:neuronType)
                    foldedNeuronIndexCount++
                }
                
                // Add bias weights to all layers except ouput
                if (index < layers.count-1)
                {
                    addNeuronToFoldedNetwork(foldedNeuronIndexCount, unfoldedIndex:foldedNeuronIndexCount, ioIndex:0, type:.Bias)
                    foldedNeuronIndexCount++
                }
            }
        }
        
        for weightString in weightStrings
        {
            let weightComponents:[String] = weightString.componentsSeparatedByString("|")
            
            let neuronComponents = weightComponents[0].componentsSeparatedByString("-")
            let valueComponents = weightComponents[1].componentsSeparatedByString(":")
            
            let fromNeuronIndex = neuronComponents[0].intValue
            let toNeuronIndex = neuronComponents[1].intValue
            let delayValue = valueComponents[1].intValue
            
            if (valueComponents[0] == "?")
            {
                self.addFoldedWeight(fromNeuronIndex, endIndex:toNeuronIndex, delay:delayValue)
            }
            else
            {
                let weightValue = valueComponents[0].doubleValue
                self.addFoldedWeight(fromNeuronIndex, endIndex:toNeuronIndex, value:weightValue, delay:delayValue)
            }
        }
        
        ////////////////////////////////////////////////////////////
        // Populate the unfolded network (unfoldedNeurons, unfoldedWeights)
        let k = self.maxFoldedWeightDelay()
        numberOfBlocks = k+1
        
        self.populateUnfoldedNetwork(k)
        self.populateTrainingStructures()
    }
    
    // k = max depth (number of blocks - 1)
    func populateUnfoldedNetwork(k:Int)
    {
        ////////////////////////////////////////////////////////////
        // First, populate the neurons of the unfolded network by repeating the nodes from the folded network
        for blockIndex in reverse(0...k)
        {
            var inputIndex:Int = 0
            var outputIndex:Int = 0
            // iterate through all neurons and append them to the unfolded network at that block
            for foldedNeuron in foldedNeurons
            {
                let foldedIndex = foldedNeuron.foldedIndex
                let neuronType = foldedNeuron.type
                let unfoldedIndex = (k-blockIndex)*foldedNeurons.count + foldedIndex
                
                var ioIndex:Int = 0
                
                switch neuronType
                {
                case .Input:
                    ioIndex = inputIndex
                case .Output:
                    ioIndex = outputIndex
                default:
                    ioIndex = 0
                }
                
                self.addNeuronToUnfoldedNetwork(foldedIndex, unfoldedIndex:unfoldedIndex, ioIndex:ioIndex, k:blockIndex, type:neuronType)
                
                switch neuronType
                {
                case .Input:
                    inputIndex++
                case .Output:
                    outputIndex++
                default:
                    break
                }
            }
        }
        
        ////////////////////////////////////////////////////////////
        // Second, assign weights in the unfolded network temporally, taking into account the delay of the folded weights
        for (fromFoldedIndex:Int, delayedWeightsFromA:[Int:[RNNDelayedWeight]]) in foldedWeights
        {
            for (toFoldedIndex:Int, delayedWeightsFromAToB:[RNNDelayedWeight]) in delayedWeightsFromA
            {
                for delayedWeight:RNNDelayedWeight in delayedWeightsFromAToB
                {
                    let delay = delayedWeight.delay
                    
                    for endBlockIndex in 0...(k-delay)
                    {
                        let startBlockIndex = endBlockIndex+delay
                        let fromUnfoldedIndex = self.unfoldedIndexForFoldedIndex(fromFoldedIndex, blockIndex:startBlockIndex)
                        let toUnfoldedIndex = self.unfoldedIndexForFoldedIndex(toFoldedIndex, blockIndex:endBlockIndex)
                        
                        self.addUnfoldedWeight(fromUnfoldedIndex, endIndex:toUnfoldedIndex, value:delayedWeight.value)
                    }
                }
            }
        }
    }
    
    func populateTrainingStructures()
    {
        for _ in 0..<unfoldedNeurons.count
        {
            outputs.append(0.0)
            deltas.append(0.0)
        }
    }
    
    //////////////////////////////
    // Modification
    //////////////////////////////
    
    func addNeuronToUnfoldedNetwork(foldedIndex:Int, unfoldedIndex:Int, ioIndex:Int, k:Int, type:NeuronType)
    {
        self.addNeuronToUnfoldedNetwork(RNNNeuron(foldedIndex:foldedIndex, unfoldedIndex:unfoldedIndex, ioIndex:ioIndex, k:k, type:type))
    }
    
    func addNeuronToUnfoldedNetwork(neuron:RNNNeuron)
    {
        unfoldedNeurons.append(neuron)
        unfoldedWeights[neuron.unfoldedIndex] = [Int:Double]()
        weightDeltas[neuron.unfoldedIndex] = [Int:Double]()
    }
    
    func addNeuronToFoldedNetwork(foldedIndex:Int, unfoldedIndex:Int, ioIndex:Int, type:NeuronType)
    {
        self.addNeuronToFoldedNetwork(RNNNeuron(foldedIndex:foldedIndex, unfoldedIndex:unfoldedIndex, ioIndex:ioIndex, k:0, type:type))
    }
    
    func addNeuronToFoldedNetwork(neuron:RNNNeuron)
    {
        foldedNeurons.append(neuron)
        foldedWeights[neuron.unfoldedIndex] = [Int:[RNNDelayedWeight]]()
    }
    
    func addFoldedWeight(startIndex:Int, endIndex:Int, delay:Int)
    {
        let smallRandomValue:Double = Double(arc4random()) / Double(UINT32_MAX)*0.1
        self.addFoldedWeight(startIndex, endIndex:endIndex, value:smallRandomValue, delay:delay)
    }
    
    func addFoldedWeight(startIndex:Int, endIndex:Int, value:Double, delay:Int)
    {
        if let weightsFromStartToEnd:[RNNDelayedWeight] = foldedWeights[startIndex]![endIndex]
        {
            foldedWeights[startIndex]![endIndex]!.append(RNNDelayedWeight(value:value, delay:delay))
        }
        else
        {
            foldedWeights[startIndex]![endIndex] = [RNNDelayedWeight(value:value, delay:delay)]
        }
    }
    
    func addUnfoldedWeight(startIndex:Int, endIndex:Int, value:Double)
    {
        if let weightsFromStart:[Int:Double] = unfoldedWeights[startIndex]
        {
            unfoldedWeights[startIndex]![endIndex] = value
            
            weightDeltas[startIndex]![endIndex] = 0.0
        }
        else
        {
            unfoldedWeights[startIndex] = [Int:Double]()
            unfoldedWeights[startIndex]![endIndex] = value
            
            weightDeltas[startIndex] = [Int:Double]()
            weightDeltas[startIndex]![endIndex] = 0.0
        }
    }
    
    //////////////////////////////
    // Information
    //////////////////////////////
    
    func unfoldedWeightPairsForFoldedWeight(fromFoldedIndex:Int, toFoldedIndex:Int, foldedWeight:RNNDelayedWeight) -> [RNNWeightIndexPair]
    {
        var unfoldedWeightPairs = [RNNWeightIndexPair]()
        let delay = foldedWeight.delay
        
        for endBlockIndex in 0...(numberOfBlocks-1-delay)
        {
            let startBlockIndex = endBlockIndex+delay
            let fromUnfoldedIndex = self.unfoldedIndexForFoldedIndex(fromFoldedIndex, blockIndex:startBlockIndex)
            let toUnfoldedIndex = self.unfoldedIndexForFoldedIndex(toFoldedIndex, blockIndex:endBlockIndex)
            
            unfoldedWeightPairs.append(RNNWeightIndexPair(startIndex:fromUnfoldedIndex, endIndex:toUnfoldedIndex))
        }
        
        return unfoldedWeightPairs
    }
    
    func unfoldedIndexForFoldedIndex(foldedIndex:Int, blockIndex:Int) -> Int
    {
        return ((numberOfBlocks - 1 - blockIndex)*foldedNeurons.count)+foldedIndex
    }
    
    func maxFoldedWeightDelay() -> Int
    {
        var maxDelay:Int = 0
        
        for (index_a:Int, weightsFromA:[Int:[RNNDelayedWeight]]) in foldedWeights
        {
            for (index_b:Int, weightsFromAToB:[RNNDelayedWeight]) in weightsFromA
            {
                for weight:RNNDelayedWeight in weightsFromAToB
                {
                    if (weight.delay > maxDelay)
                    {
                        maxDelay = weight.delay
                    }
                }
            }
        }
        
        return maxDelay
    }
    
    func neuronsConnectedToNeuron(unfoldedIndex:Int) -> [Int]
    {
        var neuronIndexes = [Int]()
        
        for (fromNeuronIndex:Int, weightsFromA:[Int:Double]) in unfoldedWeights
        {
            if let weight = weightsFromA[unfoldedIndex]
            {
                neuronIndexes.append(fromNeuronIndex)
            }
        }
        
        return neuronIndexes
    }
    
    func neuronsConnectedFromNeuron(unfoldedIndex:Int) -> [Int]
    {
        var neuronIndexes = [Int]()
        
        if let connectedNeurons:[Int:Double] = unfoldedWeights[unfoldedIndex]
        {
            for (toNeuronIndex:Int, weight:Double) in connectedNeurons
            {
                neuronIndexes.append(toNeuronIndex)
            }
        }
        
        return neuronIndexes
    }
    
    //////////////////////////////
    // Training
    //////////////////////////////
    
    func trainNetworkOnDataset(trainingSet:[RNNInstance], testSet:[RNNInstance]) -> [Double]
    {
        var accuracyOverTime = [Double]()
        
        var epochs = 0
        var previousAccuracy = 0.0
        var epochsSinceLastAccuracyChange = 0
        var noChangeToleranceThreshold = 150
        var totalEpochsThreshold = 500
        
        // Terminate if stopping conditions are met (too long without change, total epoch exceeds bounds, already at 100% accuracy)
        
        // Extra stopping conditions: epochsSinceLastAccuracyChange < noChangeToleranceThreshold &&
        while (epochs < totalEpochsThreshold && previousAccuracy < 1.0)
        {
            var totalInstancesTrained:Int = 0
            
            for instance in trainingSet
            {
                self.trainNetworkOnInstance(instance)
                totalInstancesTrained++
            }
            
            var accuracy = self.classificationAccuracy(testSet)
            accuracyOverTime.append(accuracy)
            
            if (accuracy == previousAccuracy)
            {
                epochsSinceLastAccuracyChange++
            }
            else
            {
                epochsSinceLastAccuracyChange = 0
                println("accuracy change: \(accuracy)")
            }
            
            previousAccuracy = accuracy
            
            epochs++
        }
        
        return accuracyOverTime
    }
    
    func trainNetworkOnInstance(instance:RNNInstance)
    {
        self.calculateOutputs(instance)
        self.calculateDeltas(instance)
        self.calculateWeightDeltas()
        self.applyWeightChanges()
    }
    
    func calculateOutputs(instance:RNNInstance)
    {
        for neuron in unfoldedNeurons
        {
            var output:Double = 0.0
            switch neuron.type
            {
                
            // Bias Neuron
            case .Bias:
                output = 1.0
            // Input Neuron
            case .Input:
                output = instance.inputs[neuron.k][neuron.foldedIndex]
            // Output, Hidden Neuron
            default:
                
                var net:Double = 0.0
                
                let endIndex = neuron.unfoldedIndex
                for startIndex:Int in self.neuronsConnectedToNeuron(endIndex)
                {
                    if let w:Double = unfoldedWeights[startIndex]![endIndex]
                    {
                        let o:Double = outputs[startIndex]
                        net += w*o
                    }
                }
                
                output = self.sigmoidActivation(net)
            }
            
            outputs[neuron.unfoldedIndex] = output
        }
    }
    
    func calculateDeltas(instance:RNNInstance)
    {
        for neuron in reverse(unfoldedNeurons)
        {
            var delta:Double = 0.0
            let output_j:Double = outputs[neuron.unfoldedIndex]
            
            switch neuron.type
            {
                
                // Bias Neuron
            case .Bias:
                delta = 0.0
                // Input Neuron
            case .Input:
                delta = 0.0
                // Output Neuron
            case .Output:
                delta = (instance.output[neuron.ioIndex] - output_j)*output_j*(1-output_j)
                // Hidden Neuron
            default:
                var summedDelta:Double = 0
                var connectedNeuronIndexes = self.neuronsConnectedFromNeuron(neuron.unfoldedIndex)
                
                for toIndex:Int in connectedNeuronIndexes
                {
                    // get the relevant weight
                    var weight:Double = unfoldedWeights[neuron.unfoldedIndex]![toIndex]!
                    var upperDelta:Double = deltas[toIndex]
                    
                    summedDelta += upperDelta*weight
                }
                
                delta = summedDelta*output_j*(1-output_j)
            }
            
            deltas[neuron.unfoldedIndex] = delta
        }
    }
    
    func calculateWeightDeltas()
    {
        // C*Oa*Db
        for (startIndex:Int, weightsFromA:[Int:Double]) in unfoldedWeights
        {
            for (endIndex:Int, weight_ab:Double) in unfoldedWeights[startIndex]!
            {
                let output_a = outputs[startIndex]
                let delta_b = deltas[endIndex]
                
                // NOT weight_ab, but DELTA
                weightDeltas[startIndex]![endIndex] = learningRate*output_a*delta_b
            }
        }
    }

    func applyWeightChanges()
    {
        for (fromFoldedIndex:Int, delayedWeightsFromA:[Int:[RNNDelayedWeight]]) in foldedWeights
        {
            for (toFoldedIndex:Int, delayedWeightsFromAToB:[RNNDelayedWeight]) in delayedWeightsFromA
            {
                // for every folded weight
                for delayedWeight:RNNDelayedWeight in delayedWeightsFromAToB
                {
                    // sum up the values of all the weight deltas from the corresponding unfolded weights
                    var unfoldedWeightPair = self.unfoldedWeightPairsForFoldedWeight(fromFoldedIndex, toFoldedIndex:toFoldedIndex, foldedWeight:delayedWeight)
                    
                    var accumulatedWeightDelta:Double = 0.0
                    for unfoldedWeightPair:RNNWeightIndexPair in unfoldedWeightPair
                    {
                        if let weightDelta:Double = weightDeltas[unfoldedWeightPair.startIndex]![unfoldedWeightPair.endIndex]
                        {
                            accumulatedWeightDelta += weightDelta
                        }
                    }
                    
                    for unfoldedWeightPair:RNNWeightIndexPair in unfoldedWeightPair
                    {
                        unfoldedWeights[unfoldedWeightPair.startIndex]![unfoldedWeightPair.endIndex]! += accumulatedWeightDelta
                    }
                }
            }
        }
    }
    
    func sigmoidActivation(net:Double) -> Double
    {
        return 1.0/Double(1 + pow(Double(M_E), -1*net))
    }
    
    //////////////////////////////
    // Test
    //////////////////////////////
    
    func classificationAccuracy(testSet:[RNNInstance]) -> Double
    {
        var instanceCount:Int = 0
        var correctClassificationCount:Int = 0
        
        for testInstance in testSet
        {
            if (self.testNetworkOnInstance(testInstance))
            {
                correctClassificationCount++
            }
            instanceCount++
        }
        
        return Double(correctClassificationCount)/Double(instanceCount)
    }
    
    // Pass or Fail
    func testNetworkOnInstance(instance:RNNInstance) -> Bool
    {
        let actualOutputs:[Double] = self.outputsFromNetworkOnInstance(instance)
        let expectedOutputs:[Double] = instance.output
        
        var correctClassification = true
        
        if (actualOutputs.count == expectedOutputs.count)
        {
            for index in 0..<actualOutputs.count
            {
                if (self.classificationOutput(actualOutputs[index]) != expectedOutputs[index])
                {
                    correctClassification = false
                    break
                }
            }
        }
        
        return correctClassification
    }
    
    func classificationOutput(output:Double) -> Double
    {
        var classificationOutput:Double = 0.0
        
        if (output > 0.5)
        {
            classificationOutput = 1.0
        }
        
        return classificationOutput
    }
    
    // Returns the final outputs of the network
    func outputsFromNetworkOnInstance(instance:RNNInstance) -> [Double]
    {
        var outputVector = [Double]()
        
        for neuron in unfoldedNeurons
        {
            var output:Double = 0.0
            switch neuron.type
            {
                
                // Bias Neuron
            case .Bias:
                output = 1.0
                // Input Neuron
            case .Input:
                output = instance.inputs[neuron.k][neuron.foldedIndex]
                // Output, Hidden Neuron
            default:
                
                var net:Double = 0.0
                
                let endIndex = neuron.unfoldedIndex
                for startIndex:Int in self.neuronsConnectedToNeuron(endIndex)
                {
                    if let w:Double = unfoldedWeights[startIndex]![endIndex]
                    {
                        let o:Double = outputs[startIndex]
                        net += w*o
                    }
                }
                
                output = self.sigmoidActivation(net)
            }
            
            outputs[neuron.unfoldedIndex] = output
            
            // Check to make sure this is the FINAL group of outputs (not an intermediate group in another block)
            if (neuron.k == 0 && neuron.type == .Output)
            {
                outputVector.append(output)
            }
        }
        
        return outputVector
    }
}
