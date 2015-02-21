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

class RNNNetwork
{
    var foldedNeurons = [RNNNeuron]()
    var foldedWeights = [Int:[Int:[RNNDelayedWeight]]]()
    
    var unfoldedNeurons = [RNNNeuron]()
    var unfoldedWeights = [Int:[Int:Double]]()
    
    // Training
    var outputs = [Double]()
    var deltas = [Double]()
    var weightDeltas = [Int:[Int:Double]]()
    
    init(neuronString:String, weightStrings:[String])
    {
        // Neuron String Format: number of nodes on each layer, separated by a colon "2:3:4:2"
        // Weight String Format: ["0-1|0.12:1", "1-1|0.5:0"]
        
        ////////////////////////////////////////////////////////////
        // First, populate the "folded" (abstract) network (foldedNeurons, foldedWeights)
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
        // Second, unfold the network (unfoldedNeurons, unfoldedWeights)
    }
    
    func unfoldNetwork()
    {
        
    }
    
    //////////////////////////////
    // Modification
    //////////////////////////////
    
    func addNeuronToFoldedNetwork(foldedIndex:Int, unfoldedIndex:Int, type:NeuronType)
    {
        self.addNeuronToFoldedNetwork(RNNNeuron(foldedIndex:foldedIndex, unfoldedIndex:unfoldedIndex, k:1, type:type))
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
    
    //////////////////////////////
    // Information
    //////////////////////////////
    
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
            [neuron.unfoldedIndex]
        }
    }
    
    func sigmoidActivation()
    {
        
    }
}
