# Fail Log

## Overview

This document records some possible reasons of the fail simulation that doesn't provide any log information.

## Parameter Error

If the parameter is not set correctly, the simulation may fail without any log information. For example,

```java
public class Example {
  private Map<Integer, String> map;

  public Example() {
    map.put(1, "dog");
  }
}
```

The `map` is not initialized with `new HashMap<>()`, but the compiler won't throw an error.

## Insufficient Memory

use

```java
Log.i(TAG, "Free memory before loading: " + Runtime.getRuntime().freeMemory());
```

to check the free memory before loading the yolo model. If the free memory is smaller than model size, it may cause the simulation to fail.
