<?php

namespace Codes\Josh\Neural\Basic\Mnist;

final class Data
{
    private array $samples = [];

    private array $targets = [];


    public function __construct(string $imagesPath, string $labelsPath)
    {
        $this->samples = $this->readImages($imagesPath);
        $this->targets = $this->readLabels($labelsPath);
    }


    private function readImages(string $path): array
    {
        $stream = gzopen($path, 'rb');

        $images = [];

        $header = fread($stream, 16);
        $fields = unpack("Nmagic/Nsize/Nrows/Ncols", $header);

        for ($i = 0; $i < $fields['size']; $i++) {
            $imageBytes = fread($stream, $fields['rows'] * $fields['cols']);

            $images[] = array_map(function ($b) {
                return $b / 255;
            }, array_values(unpack('C*', (string)$imageBytes)));
        }

        gzclose($stream);

        return $images;
    }


    private function readLabels(string $path): array
    {
        $stream = gzopen($path, 'rb');

        $header = fread($stream, 8);
        $fields = unpack("Nmagic/Nsize", $header);

        $labels = fread($stream, $fields['size']);

        gzclose($stream);

        return array_values(unpack('C*', (string) $labels));
    }


    public function getSamples(): array
    {
        return $this->samples;
    }


    public function getTargets(): array
    {
        return $this->targets;
    }
}
