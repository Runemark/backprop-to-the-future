//
//  RNNInstance.swift
//  Network
//
//  Created by Martin Mumford on 2/18/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Cocoa

class RNNInstance
{
    // inputs[0] = inputs at time t
    // inputs[1] = inputs at time t-1
    // inputs[2] = inputs at time t-2
    // inputs[3] = inputs at time t-3
    
    var inputs = [[Double]]()
    var output = [Double]()
    
    init(output:[Double], inputs:[[Double]])
    {
        self.inputs = inputs
        self.output = output
    }
}
