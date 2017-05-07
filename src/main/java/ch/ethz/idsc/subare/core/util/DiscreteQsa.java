// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;

public class DiscreteQsa implements QsaInterface {
  public static DiscreteQsa build(DiscreteModel discreteModel) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        tensor.append(Tensors.of(state, action));
    return new DiscreteQsa(Index.build(tensor));
  }

  private final Index index;
  private final Tensor values;

  private DiscreteQsa(Index index) {
    this.index = index;
    values = Array.zeros(index.size());
  }

  @Override
  public Scalar value(Tensor state, Tensor action) {
    return values.Get(index.of(createKey(state, action)));
  }

  @Override
  public void increment(Tensor state, Tensor action, Scalar delta) {
    values.set(scalar -> scalar.add(delta), index.of(createKey(state, action)));
  }

  private static Tensor createKey(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  public void print() {
    print(Function.identity());
  }

  public void print(Function<Scalar, Scalar> ROUND) {
    for (Tensor key : index.keys()) {
      Scalar value = values.Get(index.of(key));
      System.out.println(key + " " + value.map(ROUND));
    }
  }

  public int size() {
    return index.size();
  }
}
