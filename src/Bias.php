<?php

namespace Codes\Josh\Neural\Basic;

final class Bias
{
    private float $value;


    public function __construct(float $value)
    {
        $this->value = $value;
    }


    public function decrement(float $decrement): self
    {
        return new static($this->value - $decrement);
    }


    public function getValue(): float
    {
        return $this->value;
    }
}
