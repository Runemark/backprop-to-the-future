//
//  main.swift
//  recurrent
//
//  Created by Martin Mumford on 2/21/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

//var network:RNNNetwork = RNNNetwork.init(neuronString:"1:4:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "0-5|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "1-5|?:0", "2-7|?:0", "3-7|?:0", "4-7|?:0", "5-7|?:0", "6-7|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1"])

var network:RNNNetwork = RNNNetwork.init(neuronString:"1:2:1", weightStrings:["0-2|1.5:0", "0-3|2.0:0", "1-2|-1.0:0", "1-3|1.0:0", "2-5|2.0:0", "3-5|-2.0:0", "4-5|1.5:0", "2-2|-1.5:1", "3-3|2.0:1"])

// 5
// 2 3 4
// 0 1

var generator = ParityGenerator()
var trainingSet = generator.generateInstances(1000, desiredParity:[0,1])

network.trainNetworkOnDataset(trainingSet)

println("Hello, World!")

