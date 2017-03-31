// code by jph
package ch.ethz.idsc.subare.ch02.streets;

import java.util.List;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;

abstract class Network {
  abstract int actions();

  abstract int streets();

  abstract List<Integer> streetsFromAction(int k);

  abstract Scalar streetBias(int index);

  Tensor usage;

  final void reset() {
    usage = Array.zeros(streets());
  }

  final void feedAction(int k) {
    for (int index : streetsFromAction(k))
      usage.set(use -> use.add(RealScalar.ONE), index);
  }

  final Scalar costOfAction(int k) {
    Scalar cost = ZeroScalar.get();
    for (int index : streetsFromAction(k))
      cost = cost.add(streetBias(index).subtract(usage.get(index)));
    return cost;
  }
}
