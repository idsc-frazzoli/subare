// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.Function;
import java.util.stream.Stream;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.red.Norm;

public class DiscreteQsa implements QsaInterface {
  public static DiscreteQsa build(DiscreteModel discreteModel) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        tensor.append(Tensors.of(state, action));
    return new DiscreteQsa(Index.build(tensor), Array.zeros(tensor.length()));
  }

  private final Index index;
  private final Tensor values;

  private DiscreteQsa(Index index, Tensor values) {
    if (index.size() != values.length())
      throw new RuntimeException();
    this.index = index;
    this.values = values;
  }

  @Override
  public Scalar value(Tensor state, Tensor action) {
    return values.Get(index.of(createKey(state, action)));
  }

  @Override
  public void assign(Tensor state, Tensor action, Scalar value) {
    values.set(value, index.of(createKey(state, action)));
  }

  public DiscreteQsa create(Stream<Tensor> stream) {
    return new DiscreteQsa(index, Tensor.of(stream));
  }

  public static Tensor createKey(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  public void print() {
    print(Function.identity());
  }

  @Override
  public DiscreteQsa copy() {
    return new DiscreteQsa(index, values.copy());
  }

  public void print(Function<Scalar, Scalar> ROUND) {
    for (Tensor key : index.keys()) {
      Scalar value = values.Get(index.of(key));
      System.out.println(key + " " + value.map(ROUND));
    }
  }
  
  public Tensor values() {
    return values.unmodifiable();
  }

  public Scalar getMin() {
    return values.flatten(-1).map(Scalar.class::cast).reduce(Min::of).get();
  }

  public Scalar getMax() {
    return values.flatten(-1).map(Scalar.class::cast).reduce(Max::of).get();
  }

  public int size() {
    return index.size();
  }

  @Override
  public Scalar distance(QsaInterface vs) {
    return Norm._1.of(values.subtract(((DiscreteQsa) vs).values));
  }

  public Tensor keys() {
    return index.keys();
  }
}
