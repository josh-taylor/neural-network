<?php

namespace Codes\Josh\Neural\Basic;

final class Network
{
    /**
     * @var Layer[] $layers
     */
    private array $layers = [];

    /**
     * @var float[]|null $input
     */
    private ?array $input = null;


    /**
     * @param int $inputLength
     * @param int[] $nodesInLayer
     * @param int[] $weightsInLayer
     * @param int[] $biasInLayer
     */
    public function __construct(int $inputLength, array $nodesInLayer, array $weightsInLayer, array $biasInLayer)
    {
        if (count($nodesInLayer) !== count($weightsInLayer) || count($nodesInLayer) !== count($biasInLayer)) {
            throw new \InvalidArgumentException("Weight/Bias array lengths do not match Nodes.");
        }

        for ($i = 0; $i < count($nodesInLayer); ++$i) {
            $prevNodes = $inputLength;
            if ($i !== 0) {
                $prevNodes = $nodesInLayer[$i - 1];
            }

            $this->layers[] = new Layer($nodesInLayer[$i], $prevNodes, $weightsInLayer[$i], $biasInLayer[$i]);
        }
    }


    /**
     * @param float[] $inputValues
     *
     * @return float[]
     */
    public function calculate(array $inputValues): array
    {
        $this->input = $inputValues;
        foreach ($this->layers as $layer) {
            $inputValues = $layer->calculateValues($inputValues);
        }
        return $inputValues;
    }


    public function layer(int $index): Layer
    {
        return $this->layers[$index];
    }


    public function lastLayer(): Layer
    {
        return $this->layer($this->totalLayers() - 1);
    }


    public function totalLayers(): int
    {
        return count($this->layers);
    }


    /**
     * @return float|null
     */
    public function input(int $index): ?float
    {
        return $this->input[$index] ?? null;
    }


    /**
     * @return Layer[]
     */
    public function layers(): array
    {
        return $this->layers;
    }
}
