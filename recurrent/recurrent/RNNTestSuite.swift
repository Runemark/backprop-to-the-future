//
//  RNNTestSuite.swift
//  recurrent
//
//  Created by Martin Mumford on 2/23/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation
import AppKit

class RNNTestSuite
{
    var networksInSuite = [RNNNetwork]()
    var generator = ParityGenerator()
    
    var trainingSet:[RNNInstance]
    var testSet:[RNNInstance]
    
    var repeatLimit = 1
    
    var maxParityDepth = 0
    var desiredDepth = 0
    var dparity:[Int]
    
    init(networks:[RNNNetwork], dparity:[Int], desiredDepth:Int)
    {
        self.desiredDepth = desiredDepth
        self.dparity = dparity
        
        for network in networks
        {
            networksInSuite.append(network)
        }
        
        trainingSet = generator.generateInstances(500, desiredParity:dparity, desiredDepth:desiredDepth)
        testSet = generator.generateInstances(100, desiredParity:dparity, desiredDepth:desiredDepth)
    }
    
    func launchTestSuite()
    {
        self.maxParityDepth = max(dparity)
        
        var networkIndex = 0
        for network in networksInSuite
        {
            for repeatIndex in 0..<repeatLimit
            {
                // Clears the learning accomplished so far
                network.resetNetwork(desiredDepth)
            
                println("Initiating Network Training: Network(\(network.neuronString)) Attempt:(\(repeatIndex))")
                var accuracyOverTime = network.trainNetworkOnDataset(trainingSet, testSet:testSet)
                for accuracy in accuracyOverTime
                {
                    println("\(accuracy)")
                }
                
                // Play a glass sound after each experiment completes
                var sound = NSSound(named:"Glass")
                sound?.play()
            }
            
            networkIndex++
        }
    }
    
    func max(parityItems:[Int]) -> Int
    {
        var max:Int = 0
        
        for parity in parityItems
        {
            if (parity > max)
            {
                max = parity
            }
        }
        
        return max
    }
}
