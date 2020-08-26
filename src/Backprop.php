<?php

namespace Codes\Josh\Neural\Basic;

use InvalidArgumentException;

final class Backprop
{
    private Network $network;

    private array $expected;


    /**
     * Backprop constructor.
     *
     * @param Network $network
     * @param float[] $expected
     */
    public function __construct(Network $network, array $expected)
    {
        $this->network = $network;
        $this->expected = $expected;
    }


    /**
     * @return float[]
     */
    public function cost(): array
    {
        $result = [];
        $output = $this->network->lastLayer()->nodeActivity();

        if (count($output) !== count($this->expected)) {
            throw new InvalidArgumentException("Output length and expected length don't match.");
        }

        for ($i = 0; $i < count($output); ++$i) {
            $result[] = pow($output[$i] - $this->expected[$i], 2) / 2;
        }

        return $result;
    }


    public function totalCost(): float
    {
        return array_sum($this->cost());
    }


    public function execute(ManipulationMatrix $manipulationMatrix): void
    {
        $layerCount = $this->network->totalLayers() - 1;

        $memo = new Memo($layerCount);

        for ($layerIndex = $layerCount; $layerIndex >= 0; $layerIndex--) {
            $nodeCount = $this->network->layer($layerIndex)->totalNodes();

            for ($nodeIndex = 0; $nodeIndex < $nodeCount; ++$nodeIndex) {
                $expectedIndex = $nodeIndex;

                if ($layerIndex === $layerCount) {
                    $activityOnCost = $this->deltaActivityOnCost($layerIndex, $nodeIndex, $expectedIndex);
                    $memo->add($activityOnCost, $layerIndex, $nodeIndex);
                }

                $biasOnZ = $this->deltaBiasOnZ();
                $zOnActivity = $this->deltaZOnActivity($layerIndex, $nodeIndex);

                $activityOnCost = $memo->get($layerIndex, $nodeIndex);

                $biasOnCost = $biasOnZ * $zOnActivity * $activityOnCost;

                $manipulationMatrix->addBias($biasOnCost, $layerIndex, $nodeIndex);

                $weightCount = $this->network->layer($layerIndex)->node($nodeIndex)->totalWeights();
                for ($weightIndex = 0; $weightIndex < $weightCount; ++$weightIndex) {
                    $weightOnZ = $this->deltaWeightOnZ($layerIndex, $weightIndex);
                    $weightOnCost = $weightOnZ * $zOnActivity * $activityOnCost;

                    $manipulationMatrix->addWeight($weightOnCost, $layerIndex, $nodeIndex, $weightIndex);

                    if ($layerIndex > 0) {
                        $prevNodeIndex = $weightIndex;
                        $prevActivityOnZ = $this->deltaPrevActivityOnZ($layerIndex, $nodeIndex, $prevNodeIndex);
                        $prevActivityOnCost = $prevActivityOnZ * $zOnActivity * $activityOnCost;
                        $memo->add($prevActivityOnCost, $layerIndex - 1, $prevNodeIndex);
                    }
                }
            }
        }
    }


    private function deltaPrevActivityOnZ(int $layerIndex, int $currentNodeIndex, int $prevNodeIndex): float
    {
        return $this->network
            ->layer($layerIndex)
            ->node($currentNodeIndex)
            ->weight($prevNodeIndex)
            ->getValue();
    }


    private function deltaBiasOnZ(): int
    {
        return 1;
    }


    private function deltaWeightOnZ(int $layerIndex, int $weightIndex): float
    {
        if ($layerIndex !== 0) {
            return $this->network->layer($layerIndex - 1)->node($weightIndex)->activity();
        }

        return $this->network->input($weightIndex) ?? 0;
    }


    private function deltaZOnActivity(int $layerIndex, int $nodeIndex): float
    {
        $zValue = $this->network->layer($layerIndex)->node($nodeIndex)->zValue();
        $sig = Sigmoid::calculate($zValue);
        return ($sig * (1 - $sig));
    }


    private function deltaActivityOnCost(int $layerIndex, int $nodeIndex, int $expectedIndex): float
    {
        return $this->network->layer($layerIndex)->node($nodeIndex)->activity() - $this->expected[$expectedIndex];
    }
}
