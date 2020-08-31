<?php

namespace Codes\Josh\Neural\Basic\Mnist;

final class Image
{
    private array $input;

    private array $expectedOutput;


    public function __construct(array $input, array $expectedOutput)
    {
        $this->input = $input;
        $this->expectedOutput = $expectedOutput;
    }


    public function getInput(): array
    {
        return $this->input;
    }


    public function getExpectedOutput(): array
    {
        return $this->expectedOutput;
    }
}
