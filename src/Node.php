<?php

namespace Codes\Josh\Neural\Basic;

class Node
{
    /**
     * @var Weight[] $weights
     */
    private array $weights = [];

    private Bias $bias;

    private float $activity = 0;

    private float $zValue = 0;


    public function __construct(int $numberOfWeights, int $weightsRange, int $biasRange)
    {
        for ($i = 0; $i < $numberOfWeights; ++$i) {
            $this->weights[] = new Weight(((mt_rand() / mt_getrandmax()) - 0.5) * $weightsRange);
        }

        $this->bias = new Bias(((mt_rand() / mt_getrandmax()) - 0.5) * $biasRange);
    }


    /**
     * @param float[] $inputActivations
     *
     * @return float
     */
    public function calculateValue(array $inputActivations): float
    {
        if (count($inputActivations) !== count($this->weights)) {
            throw new \InvalidArgumentException("Length of inputs does not match length of weights.");
        }

        $sum = 0;
        for ($i = 0; $i < count($inputActivations); ++$i) {
            $sum += $inputActivations[$i] * $this->weights[$i]->getValue();
        }
        $sum += $this->bias->getValue();
        $this->zValue = $sum;

        $result = Sigmoid::calculate($sum);
        $this->activity = $result;

        return $result;
    }


    public function activity(): float
    {
        return $this->activity;
    }


    public function totalWeights(): int
    {
        return count($this->weights);
    }


    public function weight(int $index): Weight
    {
        return $this->weights[$index];
    }


    public function zValue(): float
    {
        return $this->zValue;
    }


    /**
     * @return Weight[]
     */
    public function weights(): array
    {
        return $this->weights;
    }


    public function setWeight(int $weightIndex, Weight $weight): void
    {
        $this->weights[$weightIndex] = $weight;
    }


    public function getBias(): Bias
    {
        return $this->bias;
    }


    public function setBias(Bias $bias): void
    {
        $this->bias = $bias;
    }
}
