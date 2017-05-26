# `class BasicConfig`

A human-friendly JSONable config.

Irreversible transforming to StdConfig

## `def __init__(self, snapshot)`

Args:

`snapshot`: a `util.expdir.SnapshotDirectory` object.

## Properties

Every property is a standard version.

# `class StdConfig`

There must be only one `StdConfig` object during the whole experiment.

## `def __init__(self, snapshot)`

Args:

`snapshot`: a `util.expdir.SnapshotDirectory` object.

# Standard Config

`batch_size`: `int`

`epochs`: `int`

`train_full`: `bool`

`training_loop`: `int`

`validation_loop`: `int`

`scheme`: one of `"DENSENET"`, `"TF"`, `"OTHER"` or `"NONE"`

`reg_type`: one of `"sum"` of `"mean"`

`image_size`: `int`

`layers`: a list of pairs `(layer_type, layer_config)`

```JSON
[
	"Conv",
	{
		"filters": 128,
		"kernel_size": 3,
		"stddev": ["hekaiming", {}],
		"weight_decay": ["l2", 0.0001],
		"padding": "same",
		"activation": "relu",
		"use_bn": true,
		"keep_prob": null
	}
]

[
	"Dense",
	{
		"units": 1024,
		"stddev": ["hekaiming", {}],
		"weight_decay": ["l2", 0.0001],
		"activation": "relu",
		"use_bn": true,
		"keep_prob": 0.5
	}
]

[
	"Pool",
	{
		"pool_size": 2,
		"strides": 2,
		"padding": "same",
		"keep_prob": 0.7
	}
]

[
	"Flatten",
	{}
]	
```

`minimize`: 
```JSON
[
	"momentum",
	0.1,
	{
		"momentum": 0.9,
		"use_nesterov": true
	},
	[
		"piecewise",
		{
			"boundaries": [105000, 157500],
			"values": [0.025, 0.005, 0.001]
		}
	]
]
```


