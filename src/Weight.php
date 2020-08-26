<?php

namespace Codes\Josh\Neural\Basic;

final class Weight
{
    private float $value;


    public function __construct(float $value)
    {
        $this->value = $value;
    }


    public function getValue(): float
    {
        return $this->value;
    }


    public function decrement(float $decrement): Weight
    {
        return new static($this->value - $decrement);
    }
}
