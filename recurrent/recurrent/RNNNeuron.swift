//
//  RNNNeuron.swift
//  Network
//
//  Created by Martin Mumford on 2/20/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

enum NeuronType
{
    case Output
    case Hidden
    case Input
    case Bias
}

class RNNNeuron
{
    var foldedIndex:Int
    var unfoldedIndex:Int
    var type:NeuronType
    var k:Int
    
    init(foldedIndex:Int, unfoldedIndex:Int, k:Int, type:NeuronType)
    {
        self.foldedIndex = foldedIndex
        self.unfoldedIndex = unfoldedIndex
        self.type = type
        self.k = k
    }
}