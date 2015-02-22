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
    var foldedNeurons = [RNNNeuron]()
    var foldedWeights = [Int:[Int:[RNNDelayedWeight]]]()
    
    var unfoldedNeurons = [RNNNeuron]()
    var unfoldedWeights = [Int:[Int:Double]]()
    
    var numberOfBlocks = 0
    
    // Training
    var outputs = [Double]()
    var deltas = [Double]()
    var weightDeltas = [Int:[Int:Double]]()
    
    public init(neuronString:String, weightStrings:[String])
    {
        // Neuron String Format: number of nodes on each layer, separated by a colon "2:3:4:2"
        // Weight String Format: ["0-1|0.12:1", "1-1|0.5:0"]
        
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
    
    func trainNetworkOnDataset(dataset:[RNNInstance])
    {
        var epochs:Int = 0
        var totalInstancesTrained:Int = 0
        
        for instance in dataset
        {
            self.trainNetworkOnInstance(instance)
            totalInstancesTrained++
            
            println("instances trained: \(totalInstancesTrained)")
        }
        
        epochs++
    }
    
    func trainNetworkOnInstance(instance:RNNInstance)
    {
        self.calculateOutputs(instance)
        self.calculateDeltas(instance)
        self.calculateWeightDeltas()
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
                
                output = net
            }
            
            outputs[neuron.unfoldedIndex] = self.sigmoidActivation(output)
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
                
                weightDeltas[startIndex]![endIndex] = output_a*weight_ab
            }
        }
    }
    
    func sigmoidActivation(net:Double) -> Double
    {
        return 1.0/Double(1 + pow(Double(M_E), -1*net))
    }
}
