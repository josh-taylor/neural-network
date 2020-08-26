<?php

use Codes\Josh\Neural\Basic\Backprop;
use Codes\Josh\Neural\Basic\ManipulationMatrix;
use Codes\Josh\Neural\Basic\Network;

require __DIR__ . '/vendor/autoload.php';

$network = new Network(
    2,
    [2, 2, 2],
    [1, 1, 1],
    [3, 3, 3]
);

$inputs = [];
$expectedOutputs = [];
for ($i = 0; $i < 1000; ++$i) {
    $first = round((mt_rand() / mt_getrandmax()) * 100) / 100;
    $second = round((mt_rand() / mt_getrandmax()) * 100) / 100;
    $inputs[] = [$first, $second];
    $expectedOutputs[] = [abs(1 - $first), abs(1 - $second)];
}

for ($epoch = 0; $epoch < 100; ++$epoch) {
    $epochNumber = $epoch + 1;
    echo "===== Starting Epoch #{$epochNumber} =====" . PHP_EOL;
    $epochCost = 0;
    for ($batch = 0; $batch < 100; ++$batch) {
        $manipulation = new ManipulationMatrix($network);

        for ($item = 0; $item < 10; ++$item) {
            $output = $network->calculate($inputs[$batch * 10 + $item]);
            $backprop = new Backprop($network, $expectedOutputs[$batch * 10 + $item]);
            $backprop->execute($manipulation);
            $epochCost += $backprop->totalCost();
        }

        $manipulation->apply($network, .5, .5);
    }

    echo "Total cost of epoch: ${epochCost}" . PHP_EOL;
    echo "Completed epoch #${epochNumber}" . PHP_EOL;
}

$inputs = [
    [1, 0],
    [0, 1],
    [.5, .75],
    [.25, .8],
];
$expectedOutputs = [
    [0, 1],
    [1, 0],
    [.5, .25],
    [.75, .2],
];

for ($i = 0; $i < count($inputs); ++$i) {
    $output = $network->calculate($inputs[$i]);

    echo "===== Iteration #${i} =====" . PHP_EOL;
    echo "Inputs: " . implode(',', $inputs[$i]) . PHP_EOL;
    echo "Got outputs: " . implode(',', $output) . PHP_EOL;
    echo "Expected outputs: " . implode(',', $expectedOutputs[$i]) . PHP_EOL;
}
