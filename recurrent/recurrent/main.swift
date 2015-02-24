//
//  main.swift
//  recurrent
//
//  Created by Martin Mumford on 2/21/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation
import AppKit

//var network:RNNNetwork = RNNNetwork.init(neuronString:"1:4:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "0-5|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "1-5|?:0", "2-7|?:0", "3-7|?:0", "4-7|?:0", "5-7|?:0", "6-7|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1"])

var network2:RNNNetwork = RNNNetwork.init(neuronString:"1:2:1", weightStrings:["0-2|?:0", "0-3|?:0", "1-2|?:0", "1-3|?:0", "2-5|?:0", "3-5|?:0", "4-5|?:0", "2-2|?:1", "3-3|?:1"])

var network3:RNNNetwork = RNNNetwork.init(neuronString:"1:3:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "2-6|?:0", "3-6|?:0", "4-6|?:1", "5-6|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1"])

var network4:RNNNetwork = RNNNetwork.init(neuronString:"1:4:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "0-5|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "1-5|?:0", "2-7|?:0", "2-7|?:0", "3-7|?:0", "4-7|?:0", "5-7|?:0", "6-7|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1"])

var network5:RNNNetwork = RNNNetwork.init(neuronString:"1:5:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "0-5|?:0", "0-6|?:0", "1-2|?:0", "2-3|?:0", "3-4|?:0", "4-5|?:0", "5-6|?:0", "2-8|?:0", "3-8|?:0", "4-8|?:0", "5-8|?:0", "6-8|?:0", "7-8|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1", "6-6|?:1"])

var network6:RNNNetwork = RNNNetwork.init(neuronString:"1:6:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "0-5|?:0", "0-6|?:0", "0-7|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "1-5|?:0", "1-6|?:0", "1-7|?:0", "2-9|?:0", "3-9|?:0", "4-9|?:0", "5-9|?:0", "6-9|?:0", "7-9|?:0", "8-9|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1", "6-6|?:1", "7-7|?:1"])

var network7:RNNNetwork = RNNNetwork.init(neuronString:"1:7:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "0-5|?:0", "0-6|?:0", "0-7|?:0", "0-8|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "1-5|?:0", "1-6|?:0", "1-7|?:0", "1-8|?:0", "2-10|?:0", "3-10|?:0", "4-10|?:0", "5-10|?:0", "6-10|?:0", "7-10|?:0", "8-10|?:0", "9-10|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1", "6-6|?:1", "7-7|?:1", "8-8|?:0"])

// Adding a second hidden layer becomes virtually untrainable. It requires overwhelmingly longer training times with minimal improvement. For Dparity[0,1] it is not worth the effort

var network22:RNNNetwork = RNNNetwork.init(neuronString:"1:2:2:1", weightStrings:["0-2|?:0", "0-3|?:0", "1-2|?:0", "1-3|?:0", "2-5|?:0", "2-6|?:0", "3-5|?:0", "3-6|?:0", "4-5|?:0", "4-6|?:0", "5-8|?:0", "6-8|?:0", "7-8|?:0", "2-2|?:1", "3-3|?:1", "5-5|?:1", "6-6|?:1"])
//
//var network33:RNNNetwork = RNNNetwork.init(neuronString:"1:3:3:1", weightStrings:["0-2|?:0", "0-3|?:0", "0-4|?:0", "1-2|?:0", "1-3|?:0", "1-4|?:0", "2-6|?:0", "2-7|?:0", "2-8|?:0", "3-6|?:0", "3-7|?:0", "3-8|?:0", "4-6|?:0", "4-7|?:0", "4-8|?:0", "5-6|?:0", "5-7|?:0", "5-8|?:0", "6-10|?:0", "7-10|?:0", "8-10|?:0", "9-10|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "6-6|?:1", "7-7|?:1", "8-8|?:1"])

// dparity: the inputs to be considered when generating outputs
// desired depth: how far back to actual ly look in the training instance (2 will look at t, t-1, t-2) (0 will look only at t)
var testSuite = RNNTestSuite(networks:[network4, network5, network6], dparity:[0,1,2], desiredDepth:2)
testSuite.launchTestSuite()

println("All Tasks Completed")
var sound = NSSound(named:"Submarine")
sound?.play()

