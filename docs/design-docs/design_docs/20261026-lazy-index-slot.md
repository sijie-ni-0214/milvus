# Lazy creation of sealed index slots

## Scope and configuration

Sealed segments can register persisted indexes without creating their cache
translators and slots until the index is needed. This covers scalar indexes,
including HYBRID and typed JSON projections, vector indexes, and text indexes.
Existing schema, primary-key, and index-replacement rules remain in their load
paths.

`queryNode.segcore.tieredStorage.lazyIndexSlotEnabled` is introduced in 3.0.5,
defaults to `false`, and supports configuration refresh. Lazy creation requires
the flag and an effective index warmup policy of `disable`. Configuration
updates affect subsequent load decisions; registered entries retain their
initialization policy.

The goal is to defer index metadata inspection, resource estimation, translator
construction, and slot allocation for indexes that have not been accessed.
Parameter validation and stable index registration still happen during load.

## Registration and ownership

`IndexInventory` groups entries by field and indexes each group by `IndexKey`.
Published runtime generations share entries through `shared_ptr`.

Each entry contains an immutable registration: its key, broad index category,
JSON path, and JSON cast type. JSON registration applies the existing parameter
adapter so that historical NGRAM spellings have the same category and cast as
their loaded reader. Concrete HYBRID family selection and reader capabilities
can depend on artifact metadata and are resolved on first use.

An eager entry contains its resolved family, value type, capabilities, and slot.
A lazy entry owns a factory with copied load metadata. The factory captures the
segment identifiers and other required values rather than borrowing a segment
or an in-flight load request. It produces the same resolved state as eager
loading and fixes the slot's warmup policy to `disable`.

Replacement or removal updates a new runtime generation. Older generations and
existing pins keep the entries and readers they still use. Registration reads
and retirement inspect an existing slot through `Peek()` without initializing
or waiting for an unresolved entry.

## First access and concurrency

`Entry::Resolve()` serializes initialization with a Folly distributed mutex.
Waiting callers check their own cancellation token between timed lock attempts,
at 10 ms intervals. A cancelled waiter returns `FollyCancel` independently of
an active initializer; it does not cancel the initializer or discard its factory.
Successful initialization releases the factory and its captured load metadata,
then publishes the result with an atomic release store. Readers use acquire
loads before accessing the immutable result. Concurrent runtime snapshots share
this initialization state.

An exception leaves the entry unresolved and retains its factory for a later
attempt. Each attempt uses private load state, so cancellation or failed slot
creation cannot publish a partial result. Initialization and load errors
propagate to the caller rather than selecting a raw-data fallback.

Scalar expression selection resolves the entries for its target field and
assembles a fresh `FieldIndexCapability`. It selects one complete capability
record; capabilities from different indexes are not combined. A failed entry
does not invalidate other entries that have already resolved successfully.
Vector search can locate its unique vector registration and resolve it directly
through `PinIndex()`.

`PinIndex()` opens the slot's reader when needed and checks that its capabilities
match the resolved metadata. The slot owns the reader; query consumers borrow
it for the lifetime of the pin.

## Warmup and tradeoffs

Outer slot warmup follows the configured `sync`, `async`, or `disable` policy,
including for vector engines that advertise internal `LAZY_LOAD` support. This
rule applies independently of the lazy-slot flag. Such engines retain their
non-evictable outer metadata slot; their internal loading policy remains owned
by the engine.

The first access pays deferred metadata and slot-creation work. An unresolved
entry retains a copy of its load metadata, and each capability lookup allocates
and sorts a list for the requested field. The inventory does not retain a
second capability-list cache. No fixed memory reduction or latency improvement
is assumed for a workload without measurement.

## Verification

Regression coverage exercises configuration gating, JSON registration rules,
field-scoped lookup, concurrent resolution across snapshots, failed-factory
retry, cancellation, factory release, and replacement/drop ownership. Artifact
tests cover scalar, HYBRID, dense-vector, sparse-vector, and text readers,
including disabled warmup and eager/lazy text initialization.

Cardinal regression coverage crosses lazy-slot enabled and disabled with
`disable`, `sync`, and `async` warmup. It checks first and subsequent searches,
non-evictable outer metadata, and retry using the same cold slot after a missing
artifact is restored. These fixtures use isolated local files. Community index
tests alone do not establish Cardinal behavior or production performance.

Implementation details are documented in the
[segment index architecture](../../../internal/core/src/segcore/indexing/README.md).
