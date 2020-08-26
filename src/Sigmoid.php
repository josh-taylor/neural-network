<?php

namespace Codes\Josh\Neural\Basic;

final class Sigmoid
{
    public static function calculate(float $input): float
    {
        return 1 / (1 + exp(-$input));
    }
}
