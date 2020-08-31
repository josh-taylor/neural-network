<?php

use Codes\Josh\Neural\Basic\Backprop;
use Codes\Josh\Neural\Basic\ManipulationMatrix;
use Codes\Josh\Neural\Basic\Mnist;
use Codes\Josh\Neural\Basic\Network;

require __DIR__ . '/vendor/autoload.php';

$startTime = microtime(true);

$network = new Network(
    28 * 28,
    [16, 16, 10],
    [2, 2, 2],
    [3, 3, 3]
);

echo "===== Loading Training Data =====" . PHP_EOL;

$mnist = new Mnist\Data(
    __DIR__ . '/data/digits/train-images-idx3-ubyte',
    __DIR__ . '/data/digits/train-labels-idx1-ubyte',
);

for ($trainings = 0; $trainings < 50000; ++$trainings) {
    $manipulation = new ManipulationMatrix($network);

    $setCost = 0;
    $setHits = 0;
    $setMisses = 0;

    for ($batch = 0; $batch < 10; ++$batch) {
        $randomIndex = mt_rand(0, count($mnist->getSamples()) - 1);

        $output = $network->calculate($mnist->getSamples()[$randomIndex]);

        $expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        $expected[$mnist->getTargets()[$randomIndex]] = 1;

        $backprop = new Backprop($network, $expected);

        $setCost += $backprop->totalCost();

        if (getMaxIndex($output) === ($batch % 10)) {
            $setHits++;
        } else {
            $setMisses++;
        }

        $backprop->execute($manipulation);
    }

    $manipulation->apply($network, .2, .2);

    if (($trainings + 1) % 1000 === 0 || $trainings === 0) {
        $trainingCount = $trainings + 1;
        echo "===== Training Iteration #${trainingCount} =====" . PHP_EOL;
        echo "Set Cost: " . ($setCost / 10) . PHP_EOL;
        echo "Hits vs Misses: ${setHits} / ${setMisses}" . PHP_EOL;
    }
}

if (!file_exists(__DIR__ . '/output/')) {
    mkdir(__DIR__ . '/output/', 0777);
}

file_put_contents(
    __DIR__ . '/output/number-recognition-' . time(),
    serialize($network)
);

$endTime = microtime(true);

$timeTaken = $endTime - $startTime;
$timeInMinutes = round($timeTaken / 60, 2);

echo "===== Done! =====" . PHP_EOL;
echo "Finished in {$timeInMinutes} min(s)" . PHP_EOL;

function getMaxIndex(array $array): int
{
    $maxValue = 0;
    $maxIndex = 0;
    foreach ($array as $index => $value) {
        if ($value > $maxValue) {
            $maxValue = $value;
            $maxIndex = $index;
        }
    }
    return $maxIndex;
}
