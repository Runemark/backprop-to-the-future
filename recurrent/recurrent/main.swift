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

var network5:RNNNetwork = RNNNetwork.init(neuronString:"1:5:1", weightStrings:["0-2:?:0", "0-3:?:0", "0-4:?:0", "0-5:?:0", "0-6:?:0", "1-2:?:0", "2-3:?:0", "3-4:?:0", "4-5:?:0", "5-6:?:0", "2-8|?:0", "3-8|?:0", "4-8|?:0", "5-8|?:0", "6-8|?:0", "7-8|?:0", "2-2|?:1", "3-3|?:1", "4-4|?:1", "5-5|?:1", "6-6|?:1"])

var network22:RNNNetwork = RNNNetwork.init(neuronString:"1:2:2:1", weightStrings:["0-2|?:0", "0-3|?:0", "1-2|?:0", "1-3|?:0", "2-5|?:0", "2-6|?:0", "3-5|?:0", "3-6|?:0", "4-5|?:0", "4-6|?:0", "5-8|?:0", "6-8|?:0", "7-8|?:0", "2-2|?:1", "3-3|?:1", "5-5|?:1", "6-6|?:1"])

var testSuite = RNNTestSuite(networks:[network5])
testSuite.launchTestSuite()

println("All Tasks Completed")
var sound = NSSound(named:"Submarine")
sound?.play()

