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
    
    init(networks:[RNNNetwork])
    {
        for network in networks
        {
            networksInSuite.append(network)
        }
        
        trainingSet = generator.generateInstances(500, desiredParity:[0,1])
        testSet = generator.generateInstances(100, desiredParity:[0,1])
    }
    
    func launchTestSuite()
    {
        
        
        
        var networkIndex = 2
        for network in networksInSuite
        {
            for repeatIndex in 0..<5
            {
                // Clears the learning accomplished so far
                network.resetNetwork()
                
                println("Initiating Network Training: Network(\(networkIndex)) Attempt:(\(repeatIndex))")
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
}
