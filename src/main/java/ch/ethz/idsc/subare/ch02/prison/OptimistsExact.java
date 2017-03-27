// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Supplier;
import java.util.stream.Stream;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.util.Subdivide;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.io.MathematicaFormat;

public class OptimistsExact extends AbstractExact {
  public OptimistsExact(Supplier<Agent> sup1, Supplier<Agent> sup2, int epochs) {
    super(sup1, sup2, epochs);
    // ---
    contribute(new Integer[] { 0 }, new Integer[] { 0 });
    contribute(new Integer[] { 0, 0 }, new Integer[] { 1 });
    contribute(new Integer[] { 0, 1 }, new Integer[] { 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 0, 0 });
    contribute(new Integer[] { 1 }, new Integer[] { 0, 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 1 });
  }

  public static void showOne() {
    Supplier<Agent> sup1 = //
        () -> new OptimistAgent(2, RationalScalar.of(40, 10), RationalScalar.of(10, 100));
    Supplier<Agent> sup2 = //
        () -> new OptimistAgent(2, RationalScalar.of(38, 10), RationalScalar.of(10, 100));
    OptimistsExact optimistsExact = new OptimistsExact(sup1, sup2, 200);
    System.out.println(optimistsExact.getExpectedRewards());
  }

  public static void main(String[] args) {
    Tensor init = Subdivide.range(RationalScalar.of(31, 10), RationalScalar.of(60, 10), 100);
    // System.out.println(init);
    Tensor res = Tensors.empty();
    for (Tensor c0 : init) {
      Tensor row = Tensors.empty();
      for (Tensor c1 : init) {
        Supplier<Agent> sup1 = //
            () -> new OptimistAgent(2, (Scalar) c0, RationalScalar.of(10, 100));
        Supplier<Agent> sup2 = //
            () -> new OptimistAgent(2, (Scalar) c1, RationalScalar.of(10, 100));
        OptimistsExact exact = new OptimistsExact(sup1, sup2, 50);
        row.append(Join.of(Tensors.of(c0, c1), exact.getExpectedRewards()));
      }
      res.append(row);
    }
    Stream<String> stream = MathematicaFormat.of(res);
    try {
      Files.write(Paths.get("/home/datahaki/optimist"), (Iterable<String>) stream::iterator);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}