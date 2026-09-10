package cgo

import "testing"

import "github.com/stretchr/testify/assert"

func TestCalculateReducePoolSize(t *testing.T) {
	tests := []struct {
		name               string
		maxReadConcurrency float64
		ratio              float64
		expected           int
	}{
		{name: "default", maxReadConcurrency: 8, ratio: 2, expected: 16},
		{name: "round up", maxReadConcurrency: 3, ratio: 1.5, expected: 5},
		{name: "minimum", maxReadConcurrency: 0, ratio: 0, expected: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.expected, calculateReducePoolSize(
				test.maxReadConcurrency,
				test.ratio,
			))
		})
	}
}
