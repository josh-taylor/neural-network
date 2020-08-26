<?php

namespace Codes\Josh\Neural\Basic;

final class ManipulationMatrix
{
    private array $changes = [];


    public function __construct(Network $network)
    {
        foreach ($network->layers() as $layerIndex => $layer) {
            $nodeCount = $layer->totalNodes();
            $this->changes[$layerIndex] = [];
            foreach ($layer->nodes() as $nodeIndex => $node) {
                $this->changes[$layerIndex][$nodeIndex] = ['bias' => 0.0, 'weights' => []];
            }
        }
    }


    public function addBias(float $value, int $layerIndex, int $nodeIndex): void
    {
        $this->changes[$layerIndex][$nodeIndex]['bias'] += $value;
    }


    public function addWeight(float $value, int $layerIndex, int $nodeIndex, int $weightIndex): void
    {
        if (isset($this->changes[$layerIndex][$nodeIndex]['weights'][$weightIndex])) {
            $this->changes[$layerIndex][$nodeIndex]['weights'][$weightIndex] += $value;
        } else {
            $this->changes[$layerIndex][$nodeIndex]['weights'][$weightIndex] = $value;
        }
    }


    public function export(): array
    {
        return $this->changes;
    }


    public function apply(Network $network, float $biasLearningRate, float $weightLearningRate): void
    {
        foreach ($network->layers() as $layerIndex => $layer) {
            foreach ($layer->nodes() as $nodeIndex => $node) {
                foreach ($node->weights() as $weightIndex => $weight) {
                    $weightDiff = $this->changes[$layerIndex][$nodeIndex]['weights'][$weightIndex] * $weightLearningRate;
                    $node->setWeight($weightIndex, $weight->decrement($weightDiff));
                }

                $biasDiff = $this->changes[$layerIndex][$nodeIndex]['bias'] * $biasLearningRate;
                $node->setBias($node->getBias()->decrement($biasDiff));
            }
        }
    }
}
