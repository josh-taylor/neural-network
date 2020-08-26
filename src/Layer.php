<?php

namespace Codes\Josh\Neural\Basic;

final class Layer
{
    /**
     * @var Node[] $nodes
     */
    private array $nodes = [];


    public function __construct(int $breadth, int $inputNodes, int $weightRange, int $biasRange)
    {
        for ($i = 0; $i < $breadth; $i++) {
            $this->nodes[] = new Node($inputNodes, $weightRange, $biasRange);
        }
    }


    /**
     * @param float[] $inputValues
     *
     * @return float[]
     */
    public function calculateValues(array $inputValues): array
    {
        $result = [];
        foreach ($this->nodes as $node) {
            $result[] = $node->calculateValue($inputValues);
        }
        return $result;
    }


    /**
     * @return float[]
     */
    public function nodeActivity(): array
    {
        return array_map(function (Node $node) {
            return $node->activity();
        }, $this->nodes);
    }


    public function totalNodes(): int
    {
        return count($this->nodes);
    }


    public function node(int $index): Node
    {
        return $this->nodes[$index];
    }


    /**
     * @return Node[]
     */
    public function nodes(): array
    {
        return $this->nodes;
    }
}
