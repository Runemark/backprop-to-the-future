//
//  ParityGenerator.swift
//  recurrent
//
//  Created by Martin Mumford on 2/22/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

class ParityGenerator
{
    // Parity
    func generateInstances(dataSetSize:Int, desiredParity:[Int], desiredDepth:Int) -> [RNNInstance]
    {
        var dataSet = [RNNInstance]()
        
        let maxParity = self.maxParityInSet(desiredParity)
        let stream = self.generateIOStream(dataSetSize, desiredParity:desiredParity)
        
        for index in desiredDepth..<dataSetSize+1
        {
            var instanceOutput = [Double]()
            instanceOutput.append(Double(stream.outputs[index]))
            
            var instanceInputs = [[Double]]()
            
            for timeIndex in reverse(index-desiredDepth...index)
            {
                var inputLayer = [Double]()
                inputLayer.append(Double(stream.inputs[timeIndex]))
                instanceInputs.append(inputLayer)
            }
            
            var instance:RNNInstance = RNNInstance(output:instanceOutput, inputs:instanceInputs)
            dataSet.append(instance)
        }
        
        return dataSet
    }
    
    func generateIOStream(dataSetSize:Int, desiredParity:[Int]) -> (inputs:[Int], outputs:[Int])
    {
        let maxParity = self.maxParityInSet(desiredParity)
        let necessarySize = dataSetSize + maxParity + 1
        
        var inputs = [Int]()
        var outputs = [Int]()
        
        for index in 0..<necessarySize
        {
            var randomDigit:Int = Int(arc4random_uniform(2)) // generates random number from 0 to 1
            inputs.append(randomDigit)
            
            if (index >= maxParity)
            {
                var sum = 0
                
                for parity in desiredParity
                {
                    sum += inputs[index-parity]
                }
                
                if (self.isEven(sum))
                {
                    outputs.append(0)
                }
                else
                {
                    outputs.append(1)
                }
            }
            else
            {
                
                outputs.append(-1)
            }
        }
        
        return (inputs, outputs)
    }
    
    func isEven(num:Int) -> Bool
    {
        return (num % 2 == 0)
    }
    
    func maxParityInSet(desiredParity:[Int]) -> Int
    {
        var maxParity = 0
        for parity:Int in desiredParity
        {
            if (parity > maxParity)
            {
                maxParity = parity
            }
        }
        
        return maxParity
    }
}