<?php

namespace Codes\Josh\Neural\Basic;

final class Memo
{
    private array $activationOnCost = [];


    public function __construct(int $layerCount)
    {
        $this->activationOnCost = array_fill(0, $layerCount, null);
    }


    public function add(float $value, int $layerIndex, int $nodeIndex): void
    {
        if ($this->has($layerIndex, $nodeIndex)) {
            $this->activationOnCost[$layerIndex][$nodeIndex] += $value;
        } else {
            $this->activationOnCost[$layerIndex][$nodeIndex] = $value;
        }
    }


    private function has(int $layerIndex, int $nodeIndex): bool
    {
        return isset($this->activationOnCost[$layerIndex][$nodeIndex]);
    }


    public function get(int $layerIndex, int $nodeIndex): float
    {
        if (!$this->has($layerIndex, $nodeIndex)) {
            throw new \InvalidArgumentException("No memo for activityOnCost[{$layerIndex}][$nodeIndex}].");
        }

        return $this->activationOnCost[$layerIndex][$nodeIndex];
    }
}
