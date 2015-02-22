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
                    
                    addNeuronToFoldedNetwork(foldedNeuronIndexCount, unfoldedIndex:foldedNeuronIndexCount, type:neuronType)
                    foldedNeuronIndexCount++
                }
                
                // Add bias weights to all layers except ouput
                if (index < layers.count-1)
                {
                    addNeuronToFoldedNetwork(foldedNeuronIndexCount, unfoldedIndex:foldedNeuronIndexCount, type:.Bias)
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
    }
    
    // k = max depth (number of blocks - 1)
    func populateUnfoldedNetwork(k:Int)
    {
        ////////////////////////////////////////////////////////////
        // First, populate the neurons of the unfolded network by repeating the nodes from the folded network
        for blockIndex in reverse(0...k)
        {
            // iterate through all neurons and append them to the unfolded network at that block
            for foldedNeuron in foldedNeurons
            {
                let foldedIndex = foldedNeuron.foldedIndex
                let neuronType = foldedNeuron.type
                let unfoldedIndex = (k-blockIndex)*foldedNeurons.count + foldedIndex
                
                self.addNeuronToUnfoldedNetwork(foldedIndex, unfoldedIndex:unfoldedIndex, k:blockIndex, type:neuronType)
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
    
    //////////////////////////////
    // Modification
    //////////////////////////////
    
    func addNeuronToUnfoldedNetwork(foldedIndex:Int, unfoldedIndex:Int, k:Int, type:NeuronType)
    {
        self.addNeuronToUnfoldedNetwork(RNNNeuron(foldedIndex:foldedIndex, unfoldedIndex:unfoldedIndex, k:k, type:type))
    }
    
    func addNeuronToUnfoldedNetwork(neuron:RNNNeuron)
    {
        unfoldedNeurons.append(neuron)
        unfoldedWeights[neuron.unfoldedIndex] = [Int:Double]()
    }
    
    func addNeuronToFoldedNetwork(foldedIndex:Int, unfoldedIndex:Int, type:NeuronType)
    {
        self.addNeuronToFoldedNetwork(RNNNeuron(foldedIndex:foldedIndex, unfoldedIndex:unfoldedIndex, k:0, type:type))
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
        }
        else
        {
            unfoldedWeights[startIndex] = [Int:Double]()
            unfoldedWeights[startIndex]![endIndex] = value
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
    
    //////////////////////////////
    // Training
    //////////////////////////////
    
    func calculateOutputs(instance:RNNInstance)
    {
        for neuron in unfoldedNeurons
        {
            var output:Double = 0.0
            switch neuron.type
            {
            case .Bias:
                output = 1.0
            case .Input:
                // Relies on the fact that the input nodes come first in the block
                // And thus can be accessed via the abstractIndex
                output = instance.inputs[neuron.k][neuron.foldedIndex]
            default:
                
                output = 1.0
            }
        }
    }
    
    func sigmoidActivation()
    {
        
    }
}
